# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine data-parallel axis module (Phase 1 forward-compat stub).

``plan()`` emits the same engine kwarg the translator emits for a ``dp`` axis
(``data_parallel_size``). vLLM realizes engine DP intra-engine, so this is a
:class:`DelegatedStrategy` and ``build_groups()`` / ``apply()`` are typed
no-ops. ``owned_by`` is execution-type-invariant (``vllm`` on both AR and
diffusion) but is still resolved from the backend's execution-owner table via
:func:`axis_execution_owner` so ``AxisPlan.owned_by`` has a single source of
truth for all axes (no literal that can drift from the table).
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


class DataParallelStrategy(DelegatedStrategy):
    axis = "dp"

    def __init__(self, degree: int):
        self._degree = int(degree)

    def plan(self, ctx: LoweringCtx) -> AxisPlan:
        return AxisPlan(
            axis="dp",
            degree=self._degree,
            owned_by=axis_execution_owner(ctx.backend or VLLM_BACKEND, self.axis, ctx.execution_type),
            engine_kwargs={"data_parallel_size": self._degree},
        )
