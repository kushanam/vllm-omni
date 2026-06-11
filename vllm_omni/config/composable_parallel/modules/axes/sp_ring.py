# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ring sequence-parallel axis module (Phase 1 plan-only stub).

SP is realized intra-engine by the diffusion worker's Ring runtime, so it is an
:class:`OmniExecutedStrategy`. Phase 1 ships ``plan()`` only. ``plan()`` emits
the same engine kwarg the translator emits for an ``sp_ring`` axis
(``ring_degree``), and only when the degree is > 1 — mirroring
``OmniParallelConfig.as_engine_kwargs``.
"""
from __future__ import annotations

from vllm_omni.config.composable_parallel.modules.base import (
    AxisPlan,
    LoweringCtx,
    OmniExecutedStrategy,
)


class RingSequenceParallelStrategy(OmniExecutedStrategy):
    axis = "sp_ring"

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
