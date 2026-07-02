# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline-parallel axis module (Phase 1 forward-compat stub).

``plan()`` emits the same engine kwarg the translator emits for a ``pp`` axis
(``pipeline_parallel_size``). ``build_groups()`` / ``apply()`` are the typed
no-ops inherited from :class:`DelegatedStrategy` (vLLM owns this axis).
``owned_by`` is execution-type-invariant (``vllm``) but is resolved from the
backend's execution-owner table via :func:`axis_execution_owner` so
``AxisPlan.owned_by`` has a single source of truth for all axes.
"""
from __future__ import annotations

from vllm_omni.config.composable_parallel.backends import (
    VLLM_BACKEND,
    axis_execution_owner,
)
from vllm_omni.config.composable_parallel.modules.base import (
    AxisPlan,
    DelegatedStrategy,
    LoweringCtx,
)


class PipelineParallelStrategy(DelegatedStrategy):
    axis = "pp"

    def __init__(self, degree: int):
        self._degree = int(degree)

    def plan(self, ctx: LoweringCtx) -> AxisPlan:
        return AxisPlan(
            axis="pp",
            degree=self._degree,
            owned_by=axis_execution_owner(ctx.backend or VLLM_BACKEND, self.axis, ctx.execution_type),
            engine_kwargs={"pipeline_parallel_size": self._degree},
        )
