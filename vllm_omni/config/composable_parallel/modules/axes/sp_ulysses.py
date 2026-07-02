# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ulysses sequence-parallel axis module.

Phase 1 shipped ``plan()`` only; Phase 1c (§4.5.1) lifts the SP wiring out of
``registry._apply_sequence_parallel_if_enabled`` into ``apply()`` so the
init-time dispatch loop can route SP through the same canonical
``module.apply(ctx)`` path every other axis uses.

``apply()`` is the dispatch-side entry point. Phase 1c-Twin (§3 / §5.1) it
delegates to the truly independent
:func:`vllm_omni.diffusion.distributed.sp_runtime.apply_sp_to_pipeline`
helper — the *new*-path SP wiring layer that does NOT call into the legacy
``_apply_sp_runtime`` body. The legacy OFF path (registry's thin wrapper)
keeps calling ``_apply_sp_runtime`` unchanged. Failure-mode parity with the
Phase-1b OFF path is preserved by wrapping the new-path helper call in an
identical ``try/except Exception: logger.warning(...)`` sink (N1 deferred —
§3.2 N8 / §4.5.4 row 6 / Revision log N1).
"""
from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.config.composable_parallel.backends import (
    VLLM_BACKEND,
    axis_execution_owner,
)
from vllm_omni.config.composable_parallel.modules.base import (
    ApplyCtx,
    AxisPlan,
    AxisResult,
    LoweringCtx,
    OmniExecutedStrategy,
)

logger = init_logger(__name__)


class UlyssesSequenceParallelStrategy(OmniExecutedStrategy):
    axis = "sp_ulysses"
    # Real init-time apply() (see below): dispatch this module at model init.
    supports_init_dispatch = True
    # Runs LAST of the three dispatchable axes (was APPLY_ORDER[2]); sorts after
    # sp_ring (20) so the ring-before-ulysses ordering is preserved (10<20<30).
    init_dispatch_order = 30

    def __init__(self, degree: int):
        self._degree = int(degree)

    def plan(self, ctx: LoweringCtx) -> AxisPlan:
        engine_kwargs = {"ulysses_degree": self._degree} if self._degree > 1 else {}
        return AxisPlan(
            axis="sp_ulysses",
            degree=self._degree,
            owned_by=axis_execution_owner(ctx.backend or VLLM_BACKEND, self.axis, ctx.execution_type),
            engine_kwargs=engine_kwargs,
        )

    def apply(self, ctx: ApplyCtx) -> AxisResult:
        """Dispatch-time SP wiring (§4.5.1).

        Phase 1c-Twin (§5.1): delegates to the new
        :func:`vllm_omni.diffusion.distributed.sp_runtime.apply_sp_to_pipeline`
        helper inside the SAME ``try/except Exception:
        logger.warning(...)`` sink the registry's OFF wrapper uses, so OFF
        and ON paths produce bit-identical warn-and-continue semantics on
        SP wiring failure (N1 deferred per Round-2). The new helper is the
        single place that branches on ``use_sp_descriptor``; this method
        is a thin single-arg wrapper.

        ``ctx.model`` is the just-constructed pipeline; ``ctx.od_config``
        carries the resolved :class:`DiffusionParallelConfig` from which
        the helper reads ``sequence_parallel_size`` /
        ``ulysses_degree`` / ``ring_degree`` / ``use_sp_descriptor``.
        """
        # Local import: avoids any import-time cycle with the diffusion
        # hooks package (mirrors VAE-PP's local-import pattern in vae_pp.py).
        from vllm_omni.diffusion.distributed.sp_runtime import apply_sp_to_pipeline

        try:
            applied_count = apply_sp_to_pipeline(ctx.model, ctx.od_config.parallel_config)
        except Exception as e:
            logger.warning(
                f"Failed to apply sequence parallelism: {e}. "
                "Continuing without SP hooks."
            )
            applied_count = 0
        return AxisResult(axis="sp_ulysses", hooks_applied=applied_count)
