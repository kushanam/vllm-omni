# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel axis module.

``plan()`` emits the same engine kwarg the translator emits for a ``tp`` axis
(``tensor_parallel_size``) but is execution-type-aware about *ownership*
(``REVIEW_PHASE1_IMPL`` §SHOULD-FIX 1): diffusion ``tp`` is omni-executed
(``owned_by="omni"``, ``rank_token="tp"``) per contract §5, while AR ``tp`` is
delegated to vLLM core (``owned_by="vllm"``). Only the ownership view differs —
``engine_kwargs`` is identical in both cases — so ``build_groups()`` / ``apply()``
remain the typed no-ops inherited from :class:`DelegatedStrategy` in Phase 1b.
"""
from __future__ import annotations

from vllm_omni.config.composable_parallel.modules.base import (
    AxisPlan,
    DelegatedStrategy,
    LoweringCtx,
    is_diffusion_execution,
)


class TensorParallelStrategy(DelegatedStrategy):
    axis = "tp"

    def __init__(self, degree: int):
        self._degree = int(degree)

    def plan(self, ctx: LoweringCtx) -> AxisPlan:
        if is_diffusion_execution(ctx.execution_type):
            # Diffusion TP is omni-owned (contract §5 "tp (diffusion)" row).
            return AxisPlan(
                axis="tp",
                degree=self._degree,
                owned_by="omni",
                engine_kwargs={"tensor_parallel_size": self._degree},
                rank_token="tp",
            )
        return AxisPlan(
            axis="tp",
            degree=self._degree,
            owned_by="vllm",
            engine_kwargs={"tensor_parallel_size": self._degree},
        )
