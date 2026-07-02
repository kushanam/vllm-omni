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

from typing import TYPE_CHECKING

from vllm_omni.config.composable_parallel.backends import (
    VLLM_BACKEND,
    axis_execution_owner,
)
from vllm_omni.config.composable_parallel.modules.base import (
    AxisPlan,
    DelegatedStrategy,
    LoweringCtx,
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


class DataParallelStrategy(DelegatedStrategy):
    axis = "dp"

    def __init__(self, degree: int):
        self._degree = int(degree)

    @classmethod
    def validate(cls, spec: "StrategySpec", owner: "L1Owner") -> None:
        """Validate an engine data-parallel axis.

        DP is realized intra-engine by vLLM's own DP load balancer, so it is
        engine-owned and emits no ``omni_lb_policy`` (that string configures omni's
        *replica* balancer, a different layer). Key-stable (hash) request affinity is
        not something vLLM's DP LB guarantees, so it is rejected rather than silently
        dropped.
        """
        if owner != "engine":
            _fail(
                f"dp axis {spec.name!r} is engine data parallelism realized intra-engine by "
                f"vLLM's DP load balancer; l1_owner must be 'engine', got {owner!r}. For "
                "omni-coordinator-level request fan-out across independent replicas, use a "
                "'stage_replica' axis."
            )
        if _is_affinity_dp_routing(spec.routing):
            _not_implemented(
                f"dp axis {spec.name!r} requests key-stable (hash) routing, which vLLM's "
                "intra-engine DP load balancer does not guarantee — not supported yet. Use "
                "RouteByStage(random|round_robin|least_queue) for stateless DP balancing."
            )
        if not isinstance(spec.routing, RouteByStage):
            _fail(
                f"dp axis {spec.name!r} expects RouteByStage(random|round_robin|least_queue) routing, "
                f"got {type(spec.routing).__name__}"
            )
        if spec.routing.routing_policy not in _STAGE_POLICY_TO_OMNI_LB:
            # Recognized-but-unimplemented routing (key-stable/hash) is already
            # handled above via _is_affinity_dp_routing -> _not_implemented. Anything
            # left here is an unknown/invalid policy value (e.g. a typo from YAML),
            # i.e. invalid input -> AxisTranslationError, not NotImplementedError.
            _fail(
                f"dp axis {spec.name!r} has invalid routing_policy "
                f"{spec.routing.routing_policy!r}; expected one of "
                f"{sorted(_STAGE_POLICY_TO_OMNI_LB)}."
            )

    def plan(self, ctx: LoweringCtx) -> AxisPlan:
        return AxisPlan(
            axis="dp",
            degree=self._degree,
            owned_by=axis_execution_owner(ctx.backend or VLLM_BACKEND, self.axis, ctx.execution_type),
            engine_kwargs={"data_parallel_size": self._degree},
        )
