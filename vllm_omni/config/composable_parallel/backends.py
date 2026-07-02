# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-backend axis-ownership declarations (the lowering-target property).

This module holds the framework-specific knowledge of which mesh axes a given
backend realizes **natively** (engine kwargs / deploy-level, so the composable
layer applies nothing at model init) versus **delegates** to the composable
layer's init-time ``apply()``. It replaces the vLLM-shaped assumption that was
previously baked into the global ``INIT_DISPATCHABLE`` constant in the
framework-neutral orchestrator (design §2.2).

A module is init-dispatched at model construction **iff** both opt-ins agree:

    module.supports_init_dispatch  (per-module capability, base.py §2.1)
        AND
    module.axis in backend.delegated  (per-backend delegation, here §2.2)

These are two independent safety nets: a module flag flipped early is still held
back by the backend table, and a mis-listed backend axis is still held back by
the module flag (design §3). A future non-vLLM backend ships its own
``BackendAxisOwnership`` (e.g. moving ``tp`` into ``delegated``) with zero edits
to the modules or the orchestrator.

Vocabulary note (three distinct, non-interchangeable axes of meaning):
  1. ``l1_owner`` (``engine``|``delegated``, translator) — how *routing* is realized.
  2. ``AxisPlan.owned_by`` (``omni``|``vllm``) — who *executes* the axis at runtime.
  3. backend ``native``|``delegated`` (here) — who performs the *init-time apply*.

This file is the backend-ownership home, so it now declares **two** of these three
ownership dimensions, kept lexically and semantically separate:

* ``BackendAxisOwnership.native``/``delegated`` — vocabulary **#3** (init-time
  apply). Partitions the full ``AxisName`` space.
* ``BackendAxisOwnership.executes`` (the :class:`ExecutionOwner` table) —
  vocabulary **#2** (runtime execution owner, i.e. ``AxisPlan.owned_by``). This is
  the single source of truth for who *executes* each axis, per execution regime.

They are deliberately NOT the same field, because they genuinely disagree:
diffusion ``tp`` is ``owned_by="omni"`` (vocabulary #2 — omni's diffusion engine
executes it) yet ``native`` to vLLM at init (vocabulary #3 — its module applies
nothing). Conflating them (e.g. reusing ``native``/``delegated`` to answer "who
executes") is exactly the "three-vocabulary trap" this note guards against.
Vocabulary #1 (``l1_owner``) is NOT declared here — it lives in the translator.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from vllm_omni.config.composable_parallel.modules.base import (
    AxisName,
    OwnedBy,
    is_diffusion_execution,
)


@dataclass(frozen=True)
class ExecutionOwner:
    """Who EXECUTES an axis at runtime (``AxisPlan.owned_by``), per execution regime.

    This is vocabulary **#2** (see the module docstring): the runtime execution
    owner. It is DISTINCT from :class:`BackendAxisOwnership`'s ``native``/
    ``delegated`` split (vocabulary #3 — who performs the init-time apply) and
    from the translator's ``l1_owner`` (vocabulary #1 — how routing is realized).

    Most axes are execution-type-invariant (``ar == diffusion``); only ``tp`` and
    ``ep`` flip (vLLM-executed on AR stages, omni-executed on diffusion stages).
    """

    # Owner on a non-diffusion / autoregressive stage. Also the ``None``-default
    # column: ``is_diffusion_execution(None) is False`` maps to ``ar``, preserving
    # the pre-1b AR/delegate behavior.
    ar: OwnedBy
    # Owner on a diffusion stage.
    diffusion: OwnedBy

    def resolve(self, execution_type: object | None) -> OwnedBy:
        """Resolve the runtime owner for ``execution_type``.

        Reuses the SAME predicate the axis modules used to branch on
        (:func:`is_diffusion_execution`), so ``None`` → ``ar`` (AR) and the
        result is byte-identical to the old inline branch.
        """
        return self.diffusion if is_diffusion_execution(execution_type) else self.ar


@dataclass(frozen=True)
class BackendAxisOwnership:
    """Declarative split of axes by who performs the init-time apply, per backend.

    ``native`` and ``delegated`` MUST partition the full ``AxisName`` space
    (disjoint, and together exhaustive) — enforced by the backend-exhaustiveness
    test (design §5.2). Adding a new ``AxisName`` fails that test until the
    backend table classifies it, with no central switch in the orchestrator to
    edit.

    ``executes`` is the SEPARATE vocabulary-#2 declaration (runtime execution
    owner, i.e. ``AxisPlan.owned_by``): a per-axis :class:`ExecutionOwner`, keyed
    by ``AxisName``. Only axes that have a :class:`StrategyModule` emit
    ``owned_by``, so ``executes`` is keyed on the registered module set (NOT the
    full ``AxisName`` space that ``native``/``delegated`` partition), guarded by
    its own exhaustiveness test. It defaults to an empty mapping so pre-existing
    ``BackendAxisOwnership(name, native, delegated)`` construction sites (which
    only exercise init-dispatch, vocabulary #3) stay source-compatible.
    """

    name: str
    # Axes the backend realizes itself (engine kwargs / deploy); the composable
    # layer applies NOTHING at model init for these.
    native: frozenset[AxisName]
    # Axes the backend delegates to the composable layer's init-time apply().
    delegated: frozenset[AxisName]
    # Vocabulary #2 (runtime execution owner, ``AxisPlan.owned_by``). Distinct
    # from native/delegated above (vocabulary #3). Keyed by AxisName; one entry
    # per registered StrategyModule axis.
    executes: Mapping[AxisName, ExecutionOwner] = field(default_factory=dict)


# The only backend today. Encodes vLLM's reality (design §1.4):
#   * tp/dp/pp           — real EngineArgs world dimensions (native).
#   * ep                 — a flag over TP×DP ranks (native).
#   * stage_replica      — omni deploy-level num_replicas, not init-applied (native).
#   * cfg/hsdp_*/cp      — not init-applied by the composable layer today (native).
#   * vae_pp/sp_ulysses/sp_ring — omni applies these at model init (delegated).
VLLM_BACKEND = BackendAxisOwnership(
    name="vllm",
    native=frozenset({
        "tp", "dp", "pp", "ep", "stage_replica",
        "cfg", "hsdp_shard", "hsdp_replicate", "cp",
    }),
    delegated=frozenset({"vae_pp", "sp_ulysses", "sp_ring"}),
    # Vocabulary #2 (runtime execution owner). Reproduces the module literals /
    # branches exactly (design §5, the 16-cell owned_by matrix). Only ``tp`` and
    # ``ep`` flip on execution type (vLLM on AR, omni on diffusion); the other
    # six are execution-type-invariant.
    executes={
        "tp":            ExecutionOwner(ar="vllm", diffusion="omni"),   # flips
        "ep":            ExecutionOwner(ar="vllm", diffusion="omni"),   # flips
        "dp":            ExecutionOwner(ar="vllm", diffusion="vllm"),
        "pp":            ExecutionOwner(ar="vllm", diffusion="vllm"),
        "sp_ulysses":    ExecutionOwner(ar="omni", diffusion="omni"),
        "sp_ring":       ExecutionOwner(ar="omni", diffusion="omni"),
        "stage_replica": ExecutionOwner(ar="omni", diffusion="omni"),
        "vae_pp":        ExecutionOwner(ar="omni", diffusion="omni"),
    },
)

# The canonical registry of declared backends — the single source of truth for
# "every backend that exists". A new backend is registered by adding it here;
# the backend-exhaustiveness/disjointness tests (design §5.2) parametrize over
# this tuple, so they automatically cover any future backend with no test edit.
ALL_BACKENDS: tuple[BackendAxisOwnership, ...] = (VLLM_BACKEND,)


def effective_init_dispatch_axes(
    backend: BackendAxisOwnership,
    modules: object | None = None,
) -> frozenset[AxisName]:
    """The axes actually init-dispatched for ``backend`` (capability ∩ delegated).

    The effective set is the intersection of "module supports init dispatch" and
    "backend delegates the axis" (design §2.3). For the vLLM backend this equals
    today's ``{vae_pp, sp_ulysses, sp_ring}`` — the behavior-preservation anchor
    (design §3).

    ``modules`` may be an iterable of :class:`StrategyModule` classes or instances
    (anything exposing ``axis`` / ``supports_init_dispatch``). When ``None`` it
    enumerates the canonical per-axis module classes registered under
    ``modules/axes`` — so this is a pure function of the declared modules + the
    backend table, with no hand-maintained enumeration to keep in sync.
    """
    if modules is None:
        from vllm_omni.config.composable_parallel.modules.axes import (
            STRATEGY_MODULE_CLASSES,
        )

        modules = STRATEGY_MODULE_CLASSES
    return frozenset(
        m.axis
        for m in modules
        if getattr(m, "supports_init_dispatch", False) and m.axis in backend.delegated
    )


def axis_execution_owner(
    backend: BackendAxisOwnership,
    axis: AxisName,
    execution_type: object | None,
) -> OwnedBy:
    """The runtime execution owner of ``axis`` for ``backend`` under
    ``execution_type`` (i.e. the value an axis module's ``plan()`` puts in
    ``AxisPlan.owned_by``).

    This is the SINGLE SOURCE OF TRUTH for vocabulary **#2** (who executes an
    axis at runtime). Axis modules call this instead of hard-coding an
    ``owned_by`` literal or branching on ``is_diffusion_execution`` inline, so the
    "diffusion → omni, AR → vLLM" rule for ``tp``/``ep`` (and every invariant
    axis's owner) lives in one declarative table (``backend.executes``) rather
    than being scattered across the modules.

    Distinct from :func:`effective_init_dispatch_axes` above, which answers
    vocabulary #3 (who performs the init-time apply).
    """
    # Fallback: a custom backend that declares only native|delegated (no executes
    # entry for this axis) still resolves via the historical vLLM-shaped rule,
    # which applied regardless of backend — so plan() lowering never raises a bare
    # KeyError. VLLM_BACKEND declares all 8 axes, so its path is unchanged.
    entry = backend.executes.get(axis) or VLLM_BACKEND.executes[axis]
    return entry.resolve(execution_type)
