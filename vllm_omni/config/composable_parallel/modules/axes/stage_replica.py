# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Omni stage-replica axis module (Phase 1 plan-only stub).

``stage_replica`` is an omni-coordinator-level fan-out of independent engine
replicas. It is NOT a vLLM world dimension (``consumes_world_dim=False``,
``rank_token=None``); it maps to the per-stage deploy ``num_replicas`` count plus
the pipeline-wide ``omni_lb_policy`` string. Neither is a per-stage engine arg
(``OmniParallelConfig.as_engine_kwargs`` deliberately omits both), so they are
surfaced on the plan's ``engine_kwargs`` for the module view only; the
authoritative writer remains ``apply_strategy_specs`` in Phase 1.
"""
from __future__ import annotations

from vllm_omni.config.composable_parallel.modules.base import (
    AxisPlan,
    LoweringCtx,
    OmniExecutedStrategy,
)


class StageReplicaStrategy(OmniExecutedStrategy):
    axis = "stage_replica"

    def __init__(self, degree: int, omni_lb_policy: str | None = None):
        self._degree = int(degree)
        self._omni_lb_policy = omni_lb_policy

    def plan(self, ctx: LoweringCtx) -> AxisPlan:
        engine_kwargs: dict[str, object] = {"num_replicas": self._degree}
        if self._omni_lb_policy is not None:
            engine_kwargs["omni_lb_policy"] = self._omni_lb_policy
        return AxisPlan(
            axis="stage_replica",
            degree=self._degree,
            owned_by="omni",
            engine_kwargs=engine_kwargs,
            rank_token=None,
            consumes_world_dim=False,
        )
