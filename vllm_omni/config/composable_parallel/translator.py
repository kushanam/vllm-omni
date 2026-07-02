# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate a ``StrategySpec`` stack into vLLM-Omni parallel sizing.

Read a stack of strategy specs (one per mesh axis) and work out the
tensor/data/pipeline parallel sizes, the dense-EP flag, and the number of stage
replicas, plus who owns each axis's request routing. The result is a plain,
CPU-computable sizing struct (:class:`OmniParallelConfig`) keyed by the real
``OmniEngineArgs`` / ``EngineArgs`` field names, so the deploy layer can splat it
onto a stage's engine args.

The distinction this module enforces — engine data parallelism (a true vLLM
intra-engine world dimension) vs. omni stage replicas (independent engines fanned
out by omni's coordinator) — is documented on the axis validators that depend on
it (see :meth:`DataParallelStrategy.validate` and
:meth:`StageReplicaStrategy.validate`; findings #6/#7 moved the per-axis checks
onto their modules, dispatched here via :data:`_VALIDATOR_BY_KIND`). Routing we do
not support yet (key-stable / affinity routing) raises ``NotImplementedError``;
any other invalid spec raises :class:`AxisTranslationError`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from vllm.logger import init_logger

from vllm_omni.config.composable_parallel.axis_defaults import (
    SUPPORTED_KINDS as _SUPPORTED_KINDS,
    axis_defaults,
)
from vllm_omni.config.composable_parallel.modules.axes import STRATEGY_MODULE_CLASSES
from vllm_omni.config.composable_parallel.spec import MeshAxisKind, StrategySpec
# Re-imported from the new leaf ``validation`` module (design §2.3 / ruling #1):
# these primitives moved there so the per-axis ``validate`` bodies and the
# translator can both import them cycle-free. ``_STAGE_POLICY_TO_OMNI_LB`` is
# re-exported for call-site stability (``orchestrator.py`` imports it from here);
# the translator no longer references it directly after the stage_replica
# validator moved onto its module, hence the ``# noqa: F401``.
from vllm_omni.config.composable_parallel.validation import (
    _STAGE_POLICY_TO_OMNI_LB,  # noqa: F401
    _VALID_L1_OWNERS,
    AxisTranslationError,
    L1Owner,
    _fail,
)

if TYPE_CHECKING:
    from vllm_omni.config.composable_parallel.modules.base import StrategyModule

logger = init_logger(__name__)

# DP, TP, PP, (dense) EP, stage_replica and sequence parallelism (sp_ulysses /
# sp_ring) translate today. DP/TP/PP are true world dimensions vLLM realizes
# intra-engine; dense EP is a flag over the existing TP*DP ranks; stage_replica
# is an omni-coordinator-level fan-out of independent engine replicas (NOT a
# vLLM world dimension). sp_ulysses/sp_ring shard the sequence dimension and map
# onto the diffusion engine's existing Ulysses/Ring sequence-parallel runtime
# (``ulysses_degree`` / ``ring_degree``) — no new runtime is built here, the
# declarative axes just drive the runtime that already exists. The remaining
# kinds (CFG, VAE pipelines, sparse EP, ...) need per-layer hooks or custom
# collectives and arrive in later stages.
#
# The translatable allowlist is now the derived ``SUPPORTED_KINDS`` view over the
# single ``AXIS_DEFAULTS`` table (axis_defaults.py), imported here under the
# historical private name so the gate + error message below stay byte-identical
# (it evaluates to the same tuple, in the same order, as the old literal).

# Which EngineArgs field each axis kind *sizes*. EP and stage_replica are
# deliberately absent: EP is not an independent world dimension but an
# ``enable_expert_parallel`` flag whose degree equals tp*dp; stage_replica is a
# per-stage deploy ``num_replicas`` count, not an intra-engine world dimension.
_AXIS_TO_ENGINE_FIELD: dict[MeshAxisKind, str] = {
    "tp": "tensor_parallel_size",
    "dp": "data_parallel_size",
    "pp": "pipeline_parallel_size",
}

# Default L1 owner per axis kind. ``stage_replica`` is the only delegated axis:
# omni's coordinator/StagePool owns routing across replicas. ``dp`` is engine
# data parallelism, realized by vLLM's own intra-engine DP load balancer.
_DEFAULT_L1_OWNER: dict[MeshAxisKind, L1Owner] = {
    "dp": "engine",
    "tp": "engine",
    "pp": "engine",
    "ep": "engine",
    "stage_replica": "delegated",
    # Sequence parallelism is realized intra-engine by the diffusion worker's
    # Ulysses/Ring sequence-parallel process groups, so it is engine-owned.
    "sp_ulysses": "engine",
    "sp_ring": "engine",
}

class UnmappedAxisError(AxisTranslationError):
    """Raised when an axis kind is translatable but has no ``StrategyModule``.

    The orchestrator's module-view builder must FAIL LOUDLY rather than silently
    dropping an axis that ``apply_strategy_specs`` accepted (Phase 1b fail-loud,
    ``REVIEW_PHASE1_IMPL`` §SHOULD-FIX 2). Subclassing
    :class:`AxisTranslationError` keeps existing broad translator handlers
    working while still allowing explicit ``UnmappedAxisError`` assertions.
    """


@dataclass(frozen=True)
class OmniParallelConfig:
    """Result of translating a spec stack into omni parallel sizing."""

    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    # Dense expert parallelism: a flag over the TP*DP ranks, not a world axis.
    enable_expert_parallel: bool = False
    # Number of independent omni stage replicas (the ``stage_replica`` axis).
    # This is NOT a vLLM world dimension; it is the per-stage deploy
    # ``num_replicas`` count. 1 means a single (un-replicated) engine.
    stage_replica_size: int = 1
    # The omni StagePool LB policy string for the (delegated) stage_replica
    # axis, if any. Only meaningful when ``stage_replica_size > 1``.
    omni_lb_policy: str | None = None
    # Sequence-parallel degrees. These shard the sequence dimension and map onto
    # the diffusion engine's Ulysses/Ring runtime (``ulysses_degree`` /
    # ``ring_degree``); their product is the engine's ``sequence_parallel_size``.
    # Both are true world dimensions (they consume real ranks), unlike EP.
    sp_ulysses_size: int = 1
    sp_ring_size: int = 1
    # axis kind -> resolved L1 owner ("delegated" | "engine").
    l1_owners: Mapping[MeshAxisKind, L1Owner] = field(default_factory=dict)

    @property
    def world_size(self) -> int:
        # EP and stage_replica are intentionally excluded. EP reuses the TP*DP
        # ranks rather than adding a dimension; stage_replica spins up separate
        # engines (each its own world), not extra ranks in this engine's group.
        # Sequence-parallel degrees (ulysses/ring) ARE world dimensions: they
        # consume real ranks, so [TP(4), SP_Ulysses(2)] is an 8-rank world.
        return (
            self.tensor_parallel_size
            * self.data_parallel_size
            * self.pipeline_parallel_size
            * self.sp_ulysses_size
            * self.sp_ring_size
        )

    @property
    def sequence_parallel_size(self) -> int:
        """Total SP degree (ulysses * ring), the diffusion engine's seq-parallel size."""
        return self.sp_ulysses_size * self.sp_ring_size

    @property
    def delegated_axes(self) -> tuple[MeshAxisKind, ...]:
        return tuple(kind for kind, owner in self.l1_owners.items() if owner == "delegated")

    def as_engine_kwargs(self) -> dict[str, object]:
        """Return per-stage kwargs keyed by real OmniEngineArgs/EngineArgs field names.

        Two derived values are intentionally *not* emitted here because they are
        not per-stage engine args:

        * ``stage_replica_size`` — a per-stage deploy ``num_replicas`` knob
          (``StageDeployConfig``); the deploy layer consumes it separately.
        * ``omni_lb_policy`` — a pipeline-wide load-balancer policy the engine
          reads once at construction (see ``StrategyApplyResult.omni_lb_policy``);
          it is applied at the orchestrator level, not folded into per-stage args.
        """
        kwargs: dict[str, object] = {
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
        }
        if self.enable_expert_parallel:
            kwargs["enable_expert_parallel"] = True
        # Sequence parallelism is expressed to the diffusion engine via the
        # ``ulysses_degree`` / ``ring_degree`` fields (the deploy layer nests
        # them under ``parallel_config`` and derives ``sequence_parallel_size``).
        if self.sp_ulysses_size > 1:
            kwargs["ulysses_degree"] = self.sp_ulysses_size
        if self.sp_ring_size > 1:
            kwargs["ring_degree"] = self.sp_ring_size
        return kwargs


# Kind -> the axis module class whose ``validate`` checks it, for exactly the
# translatable kinds. Derived from the two existing single-sources-of-truth —
# the registered module tuple (finding #1, ``STRATEGY_MODULE_CLASSES``) and the
# ``axis_defaults`` ``translatable`` column (finding #5) — so there is NO
# hand-maintained kind list, and "supported" (``_SUPPORTED_KINDS``) can never
# drift from "has a validator" (both filter the same ``translatable`` column).
# This resolves to exactly ``SUPPORTED_KINDS`` = {tp, dp, pp, ep, sp_ulysses,
# sp_ring, stage_replica}. ``vae_pp`` is registered but ``translatable=False``
# (the vae_pp trap, axis_defaults.py), so it is excluded — matching today's
# translator rejecting a ``vae_pp`` spec.
_VALIDATOR_BY_KIND: dict[str, type[StrategyModule]] = {
    cls.axis: cls for cls in STRATEGY_MODULE_CLASSES if axis_defaults(cls.axis).translatable
}


def _resolve_l1_owner(spec: StrategySpec) -> L1Owner:
    kind = spec.mesh_axis.kind
    declared_owner = spec.shard_extension.get("l1_owner")
    owner = declared_owner
    if owner is None:
        owner = _DEFAULT_L1_OWNER.get(kind, "engine")
        if kind not in _DEFAULT_L1_OWNER:
            logger.debug(
                "[composable_parallel] axis %r has unknown kind %r with no default l1_owner; falling back to %r",
                spec.name,
                kind,
                owner,
            )
        else:
            logger.debug(
                "[composable_parallel] axis %r (kind %r) declared no l1_owner; using default %r",
                spec.name,
                kind,
                owner,
            )
    if owner not in _VALID_L1_OWNERS:
        # Distinguish a bad value supplied via shard_extension from a bad default
        # indexed out of _DEFAULT_L1_OWNER, so the source of the invalid owner is
        # debuggable rather than swallowed.
        source = "shard_extension" if declared_owner is not None else "_DEFAULT_L1_OWNER"
        logger.debug(
            "[composable_parallel] axis %r (kind %r) resolved invalid l1_owner %r from %s; valid owners are %s",
            spec.name,
            kind,
            owner,
            source,
            sorted(_VALID_L1_OWNERS),
        )
        _fail(f"axis {kind!r} has invalid l1_owner {owner!r}; expected one of {sorted(_VALID_L1_OWNERS)}")
    return cast(L1Owner, owner)


def translate_strategy_stack(specs: Sequence[StrategySpec]) -> OmniParallelConfig:
    """Translate a spec stack into an ``OmniParallelConfig``.

    Supported kinds: dp (engine data parallel), tp, pp, (dense) ep,
    stage_replica (omni replicas), and sp_ulysses / sp_ring (sequence
    parallelism mapped onto the diffusion engine's Ulysses/Ring runtime).
    Raises ``NotImplementedError`` for deferred (affinity / key-stable)
    routing, and :class:`AxisTranslationError` for any other invalid spec (a
    kind not yet translatable, a repeated kind, an owner incompatible with the
    axis kind, or unsupported routing). The EP degree must equal
    tensor_parallel_size * data_parallel_size.
    """
    sizes: dict[str, int] = {"tensor_parallel_size": 1, "data_parallel_size": 1, "pipeline_parallel_size": 1}
    owners: dict[MeshAxisKind, L1Owner] = {}
    omni_lb_policy: str | None = None
    stage_replica_size = 1
    enable_expert_parallel = False
    ep_size: int | None = None
    sp_ulysses_size = 1
    sp_ring_size = 1

    for spec in specs:
        kind = spec.mesh_axis.kind
        if kind not in _SUPPORTED_KINDS:
            _fail(
                f"axis kind {kind!r} is not translatable yet (supported: {list(_SUPPORTED_KINDS)}); "
                "it is designed-for and lands in a later stage"
            )
        if kind in owners:
            _fail(f"axis kind {kind!r} appears more than once in the spec stack")

        owner = _resolve_l1_owner(spec)
        # Findings #6/#7 collapse: dispatch validation on the registered module
        # class for this kind — the per-kind ``if kind == ... elif`` validator
        # ladder is gone. ``kind`` is guaranteed to be a key of
        # ``_VALIDATOR_BY_KIND`` because it passed the ``_SUPPORTED_KINDS`` gate
        # above (both derive from the same ``axis_defaults`` translatable column).
        # Raise order is unchanged: supported gate -> duplicate gate -> owner
        # resolution -> ``validate``. Most validators return ``None``;
        # ``stage_replica.validate`` additionally RETURNS its resolved omni LB
        # policy string (consumed just below, exactly as the old
        # ``omni_lb_policy = _stage_replica_lb_policy(...)``).
        lb_policy = _VALIDATOR_BY_KIND[kind].validate(spec, owner)
        # The sizing byproducts each kind sets are preserved verbatim (tp/dp/pp
        # size via ``_AXIS_TO_ENGINE_FIELD`` below, unchanged). Only ep /
        # stage_replica / sp_* carry extra byproduct state, set here after the
        # validate call exactly as the old ladder did.
        if kind == "ep":
            enable_expert_parallel = True
            ep_size = spec.mesh_axis.size
        elif kind == "stage_replica":
            omni_lb_policy = lb_policy
            stage_replica_size = spec.mesh_axis.size
        elif kind == "sp_ulysses":
            sp_ulysses_size = spec.mesh_axis.size
        elif kind == "sp_ring":
            sp_ring_size = spec.mesh_axis.size

        if kind in _AXIS_TO_ENGINE_FIELD:
            sizes[_AXIS_TO_ENGINE_FIELD[kind]] = spec.mesh_axis.size
        owners[kind] = owner

    if enable_expert_parallel:
        # Dense EP shards experts across exactly the TP*DP ranks, so the declared
        # EP degree must match that product (it is not its own world dimension,
        # and PP is excluded). An EP axis of size 1 is a degenerate no-op that
        # still sets the flag; it is only valid when TP*DP == 1.
        ep_ranks = sizes["tensor_parallel_size"] * sizes["data_parallel_size"]
        if ep_size != ep_ranks:
            _fail(
                f"ep axis size {ep_size} must equal tensor_parallel_size*data_parallel_size "
                f"(={ep_ranks}); dense expert parallelism shards experts across exactly those ranks"
            )

    return OmniParallelConfig(
        tensor_parallel_size=sizes["tensor_parallel_size"],
        data_parallel_size=sizes["data_parallel_size"],
        pipeline_parallel_size=sizes["pipeline_parallel_size"],
        enable_expert_parallel=enable_expert_parallel,
        stage_replica_size=stage_replica_size,
        omni_lb_policy=omni_lb_policy,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
        l1_owners=owners,
    )
