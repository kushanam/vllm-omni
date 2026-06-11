# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense expert-parallel axis module.

Dense EP is a flag over the existing TP*DP ranks, not an independent world
dimension (``consumes_world_dim=False``). ``plan()`` emits the same engine
kwarg the translator emits for an ``ep`` axis (``enable_expert_parallel=True``)
but is execution-type-aware about *ownership* (``REVIEW_PHASE1_IMPL``
§SHOULD-FIX 1): diffusion ``ep`` is omni-executed (``owned_by="omni"``,
``rank_token="ep"``), while AR ``ep`` is delegated to vLLM core
(``owned_by="vllm"``). ``build_groups()`` / ``apply()`` are the typed no-ops
inherited from :class:`DelegatedStrategy`.

NOTE: the EP *validator* formula fix (diffusion ``ep == tp*sp*cfg*dp`` vs AR
``ep == tp*dp``, contract §6e) is deliberately NOT done here — it lands with the
EP validate step in Phase 2. This module only makes the ownership view faithful.
"""
from __future__ import annotations

from vllm_omni.config.composable_parallel.modules.base import (
    AxisPlan,
    DelegatedStrategy,
    LoweringCtx,
    is_diffusion_execution,
)


class ExpertParallelStrategy(DelegatedStrategy):
    axis = "ep"

    def __init__(self, degree: int):
        self._degree = int(degree)

    def plan(self, ctx: LoweringCtx) -> AxisPlan:
        if is_diffusion_execution(ctx.execution_type):
            # Diffusion EP is omni-owned (contract §5 "ep (diffusion)" row).
            return AxisPlan(
                axis="ep",
                degree=self._degree,
                owned_by="omni",
                engine_kwargs={"enable_expert_parallel": True},
                rank_token="ep",
                consumes_world_dim=False,
            )
        return AxisPlan(
            axis="ep",
            degree=self._degree,
            owned_by="vllm",
            engine_kwargs={"enable_expert_parallel": True},
            consumes_world_dim=False,
        )
