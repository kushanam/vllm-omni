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

from typing import TYPE_CHECKING

from vllm_omni.config.composable_parallel.backends import (
    VLLM_BACKEND,
    axis_execution_owner,
)
from vllm_omni.config.composable_parallel.modules.base import (
    AxisPlan,
    LoweringCtx,
    OmniExecutedStrategy,
)
from vllm_omni.config.composable_parallel.routing import RouteByStage
from vllm_omni.config.composable_parallel.validation import (
    _STAGE_POLICY_TO_OMNI_LB,
    _fail,
    _is_affinity_dp_routing,
    _not_implemented,
)

if TYPE_CHECKING:
    from vllm_omni.config.composable_parallel.spec import StrategySpec
    from vllm_omni.config.composable_parallel.validation import L1Owner


class StageReplicaStrategy(OmniExecutedStrategy):
    axis = "stage_replica"

    def __init__(self, degree: int, omni_lb_policy: str | None = None):
        self._degree = int(degree)
        self._omni_lb_policy = omni_lb_policy

    @classmethod
    def validate(cls, spec: "StrategySpec", owner: "L1Owner") -> str:
        """Return the omni StagePool LB policy for a delegated stage_replica axis.

        A ``stage_replica`` axis is *not* a vLLM world dimension: it stands up N
        independent engine replicas of one pipeline stage, coordinated by the omni
        coordinator and balanced over a StagePool with stateless policies only
        (random / round-robin / least-queue-length). It maps to the per-stage
        ``num_replicas`` count plus the pipeline-level ``omni_lb_policy`` string, so
        its ``l1_owner`` must be ``"delegated"`` (omni owns the routing). Key-stable
        (hash) routing is rejected because omni has no key-stable balancer yet.
        """
        if owner != "delegated":
            _fail(
                f"stage_replica axis {spec.name!r} must be 'delegated' to omni's StagePool load "
                f"balancer; got owner {owner!r}. Replica routing is owned by omni's coordinator."
            )

        routing = spec.routing
        if _is_affinity_dp_routing(routing):
            _not_implemented(
                f"stage_replica axis {spec.name!r} requests key-stable (hash) routing, which needs a "
                "dedicated load balancer — not implemented yet. Use "
                "RouteByStage(random|round_robin|least_queue) to delegate to omni's load balancer."
            )
        if not isinstance(routing, RouteByStage):
            _fail(
                f"stage_replica axis {spec.name!r} expects RouteByStage(random|round_robin|least_queue) "
                f"routing, got {type(routing).__name__}"
            )
        policy = _STAGE_POLICY_TO_OMNI_LB.get(routing.routing_policy)
        if policy is None:
            _not_implemented(
                f"stage_replica axis {spec.name!r} routing_policy {routing.routing_policy!r} has no omni LB policy"
            )
        return policy

    def plan(self, ctx: LoweringCtx) -> AxisPlan:
        engine_kwargs: dict[str, object] = {"num_replicas": self._degree}
        if self._omni_lb_policy is not None:
            engine_kwargs["omni_lb_policy"] = self._omni_lb_policy
        return AxisPlan(
            axis="stage_replica",
            degree=self._degree,
            owned_by=axis_execution_owner(ctx.backend or VLLM_BACKEND, self.axis, ctx.execution_type),
            engine_kwargs=engine_kwargs,
            rank_token=None,
            consumes_world_dim=False,
        )
