# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense expert-parallel axis module (Phase 1 forward-compat stub).

Dense EP is a flag over the existing TP*DP ranks, not an independent world
dimension (``consumes_world_dim=False``). ``plan()`` emits the same engine
kwarg the translator emits for an ``ep`` axis (``enable_expert_parallel=True``).
``build_groups()`` / ``apply()`` are the typed no-ops inherited from
:class:`DelegatedStrategy`.
"""
from __future__ import annotations

from vllm_omni.config.composable_parallel.modules.base import (
    AxisPlan,
    DelegatedStrategy,
    LoweringCtx,
)


class ExpertParallelStrategy(DelegatedStrategy):
    axis = "ep"

    def __init__(self, degree: int):
        self._degree = int(degree)

    def plan(self, ctx: LoweringCtx) -> AxisPlan:
        return AxisPlan(
            axis="ep",
            degree=self._degree,
            owned_by="vllm",
            engine_kwargs={"enable_expert_parallel": True},
            consumes_world_dim=False,
        )
