# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ring sequence-parallel axis module.

Phase 1 shipped ``plan()`` only; Phase 1c (§4.5.2) lifts the SP wiring into
``apply()`` so the init-time dispatch loop can route Ring SP through the
same canonical ``module.apply(ctx)`` path every other axis uses.

``RingSequenceParallelStrategy.apply()`` is the **idempotent companion** of
:class:`UlyssesSequenceParallelStrategy.apply` (§4.5.2): the existing SP
runtime takes a SINGLE ``SequenceParallelConfig`` carrying both
``ulysses_degree`` and ``ring_degree``, so the wiring is one logical
application even in hybrid (Ulysses+Ring) configs. To avoid double-registering
hooks under the dispatch loop when both modules are present, ``apply()``
short-circuits at the dispatch boundary if
:attr:`ForwardContext.sp_plan_hooks_applied` is already True. Otherwise (the
ring-only case) it performs the full SP wiring identically to ``sp_ulysses``.

Phase 1c-Twin (§3 / §5.2): the new-path helper this module dispatches to is
:func:`vllm_omni.diffusion.distributed.sp_runtime.apply_sp_to_pipeline`,
truly independent of the legacy ``_apply_sp_runtime`` body (which still
backs the OFF path).

Failure-mode parity with the Phase-1b OFF path is preserved by wrapping the
new-path helper call in an identical ``try/except Exception:
logger.warning(...)`` sink (N1 deferred — §3.2 N8 / §4.5.4 row 6 / Revision
log N1).
"""
from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.config.composable_parallel.modules.base import (
    ApplyCtx,
    AxisPlan,
    AxisResult,
    LoweringCtx,
    OmniExecutedStrategy,
)

logger = init_logger(__name__)


class RingSequenceParallelStrategy(OmniExecutedStrategy):
    axis = "sp_ring"
    # Real init-time apply() (see below): dispatch this module at model init.
    supports_init_dispatch = True
    # Runs BEFORE sp_ulysses (was APPLY_ORDER[1]). Rationale: the ring-only case
    # (no Ulysses) reaches the shared SP runtime helper through this module's
    # apply() first; the Ulysses-or-hybrid case still produces a single helper
    # application thanks to the SP-side idempotency check (orchestrator §4.5.2).
    init_dispatch_order = 20

    def __init__(self, degree: int):
        self._degree = int(degree)

    def plan(self, ctx: LoweringCtx) -> AxisPlan:
        engine_kwargs = {"ring_degree": self._degree} if self._degree > 1 else {}
        return AxisPlan(
            axis="sp_ring",
            degree=self._degree,
            owned_by="omni",
            engine_kwargs=engine_kwargs,
        )

    def apply(self, ctx: ApplyCtx) -> AxisResult:
        """Dispatch-time SP wiring — idempotent companion (§4.5.2).

        If a previous step (e.g. ``sp_ulysses.apply()`` in a hybrid config,
        or a re-entry of this same loop) already registered SP hooks this
        stage init — signalled by
        :attr:`ForwardContext.sp_plan_hooks_applied` being True — return
        immediately with ``hooks_applied=0`` and a note explaining the
        no-op. Otherwise delegate (Phase 1c-Twin §5.2) to the new
        :func:`vllm_omni.diffusion.distributed.sp_runtime.apply_sp_to_pipeline`
        helper inside the SAME ``try/except Exception:
        logger.warning(...)`` sink ``sp_ulysses`` and the registry's OFF
        wrapper use, so OFF and ON paths produce bit-identical
        warn-and-continue semantics on SP wiring failure (N1 deferred per
        Round-2).

        The new helper itself ALSO short-circuits on the marker (§5.5 A4
        re-entry semantics); this dispatch-boundary check is the
        belt-and-braces guarantee the spec calls for in the §4.5.2
        "idempotent companion" contract.
        """
        # Local imports keep the module's import graph cycle-free with
        # the diffusion hooks / forward_context modules.
        from vllm_omni.diffusion.distributed.sp_runtime import apply_sp_to_pipeline
        from vllm_omni.diffusion.forward_context import get_forward_context

        fc = get_forward_context()
        if fc.sp_plan_hooks_applied:
            return AxisResult(
                axis="sp_ring",
                hooks_applied=0,
                notes=("SP already applied; sp_ring is idempotent companion",),
            )

        try:
            applied_count = apply_sp_to_pipeline(ctx.model, ctx.od_config.parallel_config)
        except Exception as e:
            logger.warning(
                f"Failed to apply sequence parallelism: {e}. "
                "Continuing without SP hooks."
            )
            applied_count = 0
        return AxisResult(axis="sp_ring", hooks_applied=applied_count)
