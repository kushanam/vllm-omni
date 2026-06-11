# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ulysses sequence-parallel axis module (Phase 1 plan-only stub).

SP is realized intra-engine by the diffusion worker's Ulysses runtime, so it is
an :class:`OmniExecutedStrategy`. Phase 1 ships ``plan()`` only (the group build
+ hook wiring land in Phase 1b). ``plan()`` emits the same engine kwarg the
translator emits for an ``sp_ulysses`` axis (``ulysses_degree``), and only when
the degree is > 1 — mirroring ``OmniParallelConfig.as_engine_kwargs``.
"""
from __future__ import annotations

from vllm_omni.config.composable_parallel.modules.base import (
    AxisPlan,
    LoweringCtx,
    OmniExecutedStrategy,
)


class UlyssesSequenceParallelStrategy(OmniExecutedStrategy):
    axis = "sp_ulysses"

    def __init__(self, degree: int):
        self._degree = int(degree)

    def plan(self, ctx: LoweringCtx) -> AxisPlan:
        engine_kwargs = {"ulysses_degree": self._degree} if self._degree > 1 else {}
        return AxisPlan(
            axis="sp_ulysses",
            degree=self._degree,
            owned_by="omni",
            engine_kwargs=engine_kwargs,
        )
