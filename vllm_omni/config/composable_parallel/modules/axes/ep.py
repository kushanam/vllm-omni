# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense expert-parallel axis module.

Dense EP is a flag over the existing TP*DP ranks, not an independent world
dimension (``consumes_world_dim=False``). ``plan()`` emits the same engine
kwarg the translator emits for an ``ep`` axis (``enable_expert_parallel=True``)
but is execution-type-aware about *ownership* (``REVIEW_PHASE1_IMPL``
§SHOULD-FIX 1): diffusion ``ep`` is omni-executed (``owned_by="omni"``,
``rank_token="ep"``), while AR ``ep`` is delegated to vLLM core
(``owned_by="vllm"``). That backend-shaped rule is resolved from the backend's
execution-owner table via :func:`axis_execution_owner` (the single source of
truth, vocabulary #2) rather than an inline ``is_diffusion_execution`` branch.
``build_groups()`` / ``apply()`` are the typed no-ops inherited from
:class:`DelegatedStrategy`.

NOTE: the EP *validator* formula fix (diffusion ``ep == tp*sp*cfg*dp`` vs AR
``ep == tp*dp``, contract §6e) is deliberately NOT done here — it lands with the
EP validate step in Phase 2. This module only makes the ownership view faithful.
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


class ExpertParallelStrategy(DelegatedStrategy):
    axis = "ep"

    def __init__(self, degree: int):
        self._degree = int(degree)

    def plan(self, ctx: LoweringCtx) -> AxisPlan:
        # Owner from the backend execution-owner table (vocabulary #2), not an
        # inline branch. ``rank_token`` is "ep" only when omni executes it; None
        # when vLLM owns it — reproduces the pre-refactor value.
        owner = axis_execution_owner(ctx.backend or VLLM_BACKEND, self.axis, ctx.execution_type)
        return AxisPlan(
            axis="ep",
            degree=self._degree,
            owned_by=owner,
            engine_kwargs={"enable_expert_parallel": True},
            rank_token="ep" if owner == "omni" else None,
            consumes_world_dim=False,
        )
