# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StrategyModule contract types (Phase 1, §2).

The ``plan()`` path is torch-free: it runs on CPU inside ``StageConfigFactory``.
torch only appears in ``GroupBuildCtx`` / ``ApplyCtx`` and
``GroupHandle.coordinator``, all guarded under ``TYPE_CHECKING``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from torch.distributed import ProcessGroup
    # omni GroupCoordinator (diffusion side). Imported lazily to keep plan() torch-free.
    from vllm_omni.diffusion.distributed.parallel_state import GroupCoordinator
    # Backend axis-ownership table (annotation only). Imported under TYPE_CHECKING
    # so base.py stays runtime-independent of backends.py (which imports base.py),
    # keeping the module import graph cycle-free.
    from vllm_omni.config.composable_parallel.backends import BackendAxisOwnership
    # Spec + l1_owner types for the ``validate`` contract (annotation only).
    # Imported under TYPE_CHECKING so base.py stays a leaf w.r.t. spec.py /
    # validation.py at runtime (validation.py does NOT import base.py, so this is
    # cycle-free even for the type checker).
    from vllm_omni.config.composable_parallel.spec import StrategySpec
    from vllm_omni.config.composable_parallel.validation import L1Owner

# ---------------------------------------------------------------------------
# Axis identity
# ---------------------------------------------------------------------------
AxisName = Literal[
    "tp", "dp", "pp", "ep",
    "sp_ulysses", "sp_ring",
    "cfg",
    "hsdp_shard", "hsdp_replicate",   # split from today's single "hsdp" (contract §6d)
    "vae_pp",
    "stage_replica",
    "cp",
]

OwnedBy = Literal["omni", "vllm"]


def is_diffusion_execution(execution_type: object | None) -> bool:
    """True iff a stage's execution type marks it as a diffusion stage.

    Used by execution-type-sensitive axis modules (``tp`` / ``ep``) to decide
    ownership: diffusion ``tp`` / ``ep`` are omni-executed, whereas AR ``tp`` /
    ``ep`` are delegated to vLLM core (``REVIEW_PHASE1_IMPL`` §SHOULD-FIX 1).

    Compares by the enum *value* string (``"diffusion"``) to stay torch-free and
    avoid importing the heavy ``stage_config`` module into the ``plan()`` path.
    This matches both ``StageExecutionType.DIFFUSION`` and the legacy
    ``StageType.DIFFUSION`` (their values are identical). ``None`` (no signal,
    e.g. a caller that does not thread execution type) defaults to non-diffusion,
    preserving the pre-1b AR/delegate behavior.
    """
    if execution_type is None:
        return False
    return getattr(execution_type, "value", execution_type) == "diffusion"


# ---------------------------------------------------------------------------
# Carriers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AxisPlan:
    """CPU-only result of plan(). Pure data — no torch, no process groups."""
    axis: AxisName
    degree: int                          # this axis's size
    owned_by: OwnedBy                    # who EXECUTES it ("omni" | "vllm").
    #   ORTHOGONAL to the translator's l1_owner ("engine"|"delegated", i.e. HOW
    #   routing is realized): e.g. diffusion SP is l1_owner="engine" (applied as
    #   ulysses_degree/ring_degree kwargs) yet owned_by="omni" (omni's diffusion
    #   engine executes it, not vLLM core). Do not conflate the two vocabularies.
    engine_kwargs: Mapping[str, object] = field(default_factory=dict)
    #   delegated/sizing kwargs to splat into a stage's engine args, keyed by the
    #   real EngineArgs/OmniEngineArgs field names (e.g. {"tensor_parallel_size": 4},
    #   {"vae_patch_parallel_size": 2}). Exactly the today-emitted shape (translator
    #   as_engine_kwargs / DiffusionParallelConfig field names).
    rank_token: str | None = None        # RankGenerator token ("tp"/"sp"/"pp"/"cfg"/
    #   "dp"/"fs"/"ep"); None when the axis is not a DiT-world dimension (e.g.
    #   stage_replica, vae_pp — vae_pp REUSES the DiT group, see §4.1).
    consumes_world_dim: bool = True      # True iff degree multiplies the per-replica
    #   device count (drives world_size + device-layout validation). vae_pp and
    #   stage_replica are False.
    requires: frozenset[str] = frozenset()   # model capabilities needed (e.g. {"sp_descriptor"})


@dataclass(frozen=True)
class GroupHandle:
    """Reference to a process group owned by omni, or a typed marker.

    NOTE (§8 G2): coordinator may be a real omni GroupCoordinator, a *raw*
    torch ProcessGroup (the DiT group is built via torch.distributed.new_group,
    not a GroupCoordinator), or None when delegated / reused.
    """
    axis: AxisName
    parallel_mode: str                   # "tensor"/"sequence"/"pipeline"/
    #   "classifier_free_guidance"/"data"/"fully_shard"/"expert"/"delegated"/"reused"
    coordinator: "GroupCoordinator | ProcessGroup | None" = None
    ranks: tuple[tuple[int, ...], ...] = ()
    delegated: bool = False              # True ⇒ vLLM created/owns this group
    reused: bool = False                 # True ⇒ omni reuses an existing group (vae_pp → DiT)


@dataclass(frozen=True)
class AxisResult:
    """Result of build_groups()/apply(). Threaded back to the orchestrator."""
    axis: AxisName
    group: GroupHandle | None = None     # None for apply()-only steps
    hooks_applied: int = 0               # # model hooks registered (SP); 0 for delegates
    notes: tuple[str, ...] = ()

    @classmethod
    def delegated(cls, axis: AxisName) -> "AxisResult":
        return cls(axis=axis, group=GroupHandle(axis, "delegated", None, (), delegated=True))


# ---------------------------------------------------------------------------
# Lowering / build / apply contexts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoweringCtx:
    """CPU context for plan(). Carries the parsed front-end + stage signal."""
    spec: object | None = None           # the StrategySpec for this axis (yaml front-end), or None
    raw_degree: int | None = None        # degree from raw deploy/engine args (no-yaml front-end)
    execution_type: object | None = None # StageExecutionType (disambiguates tp owner; §8 N1)
    shard_extension: Mapping[str, object] = field(default_factory=dict)
    # The active backend's axis-ownership table, consulted by execution-type-
    # sensitive modules (tp/ep) — and, for single-source-of-truth, every axis
    # module — to resolve ``AxisPlan.owned_by`` via ``axis_execution_owner``.
    # ``None`` means "the default vLLM backend" (resolved lazily by the module,
    # avoiding an import-time dependency on backends.py), so bare
    # ``plan(LoweringCtx())`` introspection calls stay behavior-identical.
    backend: "BackendAxisOwnership | None" = None


@dataclass
class GroupBuildCtx:
    """GPU context for build_groups() (worker init). torch present."""
    rank_generator: object               # the orchestrator-owned RankGenerator
    backend: str
    world_size: int


@dataclass
class ApplyCtx:
    """GPU context for apply() (model init). torch present.

    Phase 1c adds forward-compat fields (``execution_type``, ``device``,
    ``rank``, ``group_handles``) used by the init-time dispatch loop in
    ``Orchestrator.apply``; all default to ``None`` / ``{}`` so existing
    Phase-1 / Phase-1b construction sites stay source-compatible.
    """
    model: "nn.Module"
    od_config: object                    # OmniDiffusionConfig (carries parallel_config)
    # StageExecutionType for this stage; lets execution-type-sensitive axis
    # modules (e.g. ``tp`` / ``ep``) decide ownership at apply time without
    # having to re-derive it from od_config. None preserves the pre-1c shape.
    execution_type: object | None = None
    # torch.device of the current rank. Forward-compat for axes that need
    # device pinning (HSDP-into-dispatch, contract §6d). None today.
    device: "torch.device | None" = None
    # Current rank within the world group. Forward-compat for axes that need
    # the current rank for warning-once / rank-0 logging. None today.
    rank: int | None = None
    # Per-axis GroupHandles. Empty in Phase 1c; Phase 2 fills it from
    # ``build_groups()`` results so apply() can consume the typed handles
    # produced by group construction. SP and VAE-PP do not consume this
    # in Phase 1c — they read ``get_sp_group()`` directly / reuse the DiT
    # group lazily.
    group_handles: dict[AxisName, GroupHandle] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The ONE interface
# ---------------------------------------------------------------------------
@runtime_checkable
class StrategyModule(Protocol):
    axis: AxisName
    # Per-module init-dispatch capability (§2.1). True iff this module performs
    # real init-time ``apply()`` work that the orchestrator should run at model
    # construction. Co-located with the ``apply()`` body it guards; default
    # False (declared on the base classes) so a new axis is inert until its
    # author explicitly opts in. Replaces the old global ``INIT_DISPATCHABLE``.
    supports_init_dispatch: bool
    # Ascending order key for the init-dispatch loop (§2.3). The orchestrator
    # sorts dispatchable modules by ``(init_dispatch_order, axis)``. Only
    # meaningful when ``supports_init_dispatch`` is True. Replaces the old
    # global ``APPLY_ORDER`` tuple.
    init_dispatch_order: int
    # Per-axis translator validation (findings #6/#7). A classmethod so the
    # translator can dispatch on the module *class* without constructing an
    # instance (construction needs the degree, and ``stage_replica`` needs the
    # omni LB policy — neither is needed to validate). ``owner`` is the
    # translator's resolved ``l1_owner`` (vocabulary #1), NOT ``AxisPlan.owned_by``.
    # Most axes return ``None``; ``stage_replica`` additionally RETURNS its
    # resolved omni LB policy string (the translator loop consumes the return
    # value), hence the ``str | None`` return.
    @classmethod
    def validate(cls, spec: "StrategySpec", owner: "L1Owner") -> str | None: ...
    def plan(self, ctx: LoweringCtx) -> AxisPlan: ...
    def build_groups(self, ctx: GroupBuildCtx) -> AxisResult: ...
    def apply(self, ctx: ApplyCtx) -> AxisResult: ...


class OmniExecutedStrategy:
    """Base for axes omni executes. Subclasses implement all three methods."""
    axis: AxisName
    # Default: not init-applied. Modules with a real ``apply()`` body flip this
    # True next to that ``apply()`` (e.g. vae_pp / sp_ulysses / sp_ring). Modules
    # that inherit the NotImplementedError ``apply()`` (e.g. stage_replica) leave
    # it False — they have no init-time work.
    supports_init_dispatch: bool = False
    # Ascending init-dispatch order key (§2.3); unused while
    # ``supports_init_dispatch`` is False.
    init_dispatch_order: int = 0
    # Default: no translator validation. Translatable axes override this with the
    # verbatim per-axis check (findings #6/#7); a module that is dispatched by the
    # translator without overriding it fails loudly rather than silently passing.
    @classmethod
    def validate(cls, spec: "StrategySpec", owner: "L1Owner") -> str | None: raise NotImplementedError
    def plan(self, ctx: LoweringCtx) -> AxisPlan: raise NotImplementedError
    def build_groups(self, ctx: GroupBuildCtx) -> AxisResult: raise NotImplementedError
    def apply(self, ctx: ApplyCtx) -> AxisResult: raise NotImplementedError


class DelegatedStrategy:
    """Base for axes vLLM executes. plan() is real; build/apply are TYPED no-ops."""
    axis: AxisName
    owned_by: OwnedBy = "vllm"
    # The backend realizes these axes natively (engine kwargs / deploy); the
    # composable layer never applies them at init, so this is always False.
    supports_init_dispatch: bool = False
    init_dispatch_order: int = 0
    # Default: no translator validation (see OmniExecutedStrategy.validate).
    @classmethod
    def validate(cls, spec: "StrategySpec", owner: "L1Owner") -> str | None: raise NotImplementedError
    def plan(self, ctx: LoweringCtx) -> AxisPlan: raise NotImplementedError
    def build_groups(self, ctx: GroupBuildCtx) -> AxisResult:
        return AxisResult.delegated(self.axis)
    def apply(self, ctx: ApplyCtx) -> AxisResult:
        return AxisResult.delegated(self.axis)
