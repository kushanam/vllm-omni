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
The new concept is specifically about model-init application. It is close to
``owned_by`` but not identical: diffusion ``tp`` is ``owned_by="omni"`` yet
``native`` to vLLM at init (its module applies nothing) — which is exactly why a
separate declaration is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.config.composable_parallel.modules.base import AxisName


@dataclass(frozen=True)
class BackendAxisOwnership:
    """Declarative split of axes by who performs the init-time apply, per backend.

    ``native`` and ``delegated`` MUST partition the full ``AxisName`` space
    (disjoint, and together exhaustive) — enforced by the backend-exhaustiveness
    test (design §5.2). Adding a new ``AxisName`` fails that test until the
    backend table classifies it, with no central switch in the orchestrator to
    edit.
    """

    name: str
    # Axes the backend realizes itself (engine kwargs / deploy); the composable
    # layer applies NOTHING at model init for these.
    native: frozenset[AxisName]
    # Axes the backend delegates to the composable layer's init-time apply().
    delegated: frozenset[AxisName]


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
