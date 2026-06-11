# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel axis module (Phase 1 forward-compat stub).

``plan()`` emits the same engine kwarg the translator emits for a ``tp`` axis
(``tensor_parallel_size``). ``build_groups()`` / ``apply()`` are the typed
no-ops inherited from :class:`DelegatedStrategy` (vLLM owns this axis).
"""
from __future__ import annotations

from vllm_omni.config.composable_parallel.modules.base import (
    AxisPlan,
    DelegatedStrategy,
    LoweringCtx,
)


class TensorParallelStrategy(DelegatedStrategy):
    axis = "tp"

    def __init__(self, degree: int):
        self._degree = int(degree)

    def plan(self, ctx: LoweringCtx) -> AxisPlan:
        return AxisPlan(
            axis="tp",
            degree=self._degree,
            owned_by="vllm",
            engine_kwargs={"tensor_parallel_size": self._degree},
        )
