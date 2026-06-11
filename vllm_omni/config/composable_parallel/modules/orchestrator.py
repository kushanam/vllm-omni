# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-1 orchestrator: a thin completion of the existing translate/apply path.

Coexistence strategy (§3.1, "wrap, don't reimplement"): ``lower_and_plan`` calls
the EXISTING :func:`apply_strategy_specs` (which itself calls
``translate_strategy_stack``) unchanged and returns its ``StrategyApplyResult``
verbatim as the byte-identical overlay source. It *additionally* constructs a
parallel module view (a list of :class:`StrategyModule` per role, with each
module's ``plan()`` decomposed) for introspection / forward-compat. The module
view is NOT yet the overlay writer in Phase 1; the §5.2 equivalence test asserts
the aggregate of the module plans reproduces the monolithic translator output,
which lets Phase 2 promote the module view to the writer safely.

The GPU phases (``build_groups`` / ``apply``) are skeletons here; their bodies
land in Phase 2+.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.config.composable_parallel.apply import (
    StrategyApplyResult,
    apply_strategy_specs,
)
from vllm_omni.config.composable_parallel.modules.axes import (
    DataParallelStrategy,
    ExpertParallelStrategy,
    PipelineParallelStrategy,
    RingSequenceParallelStrategy,
    StageReplicaStrategy,
    TensorParallelStrategy,
    UlyssesSequenceParallelStrategy,
)
from vllm_omni.config.composable_parallel.modules.base import (
    AxisPlan,
    LoweringCtx,
    StrategyModule,
)
from vllm_omni.config.composable_parallel.translator import _STAGE_POLICY_TO_OMNI_LB

StrategyPlan = list[StrategyModule]

# Mesh-axis kind -> the module factory that builds it from a StrategySpec. The
# delegate axes (tp/dp/pp/ep) and SP axes take only the declared degree; the
# stage_replica module additionally needs the resolved omni LB policy.
_MODULE_BY_KIND = {
    "tp": TensorParallelStrategy,
    "dp": DataParallelStrategy,
    "pp": PipelineParallelStrategy,
    "ep": ExpertParallelStrategy,
    "sp_ulysses": UlyssesSequenceParallelStrategy,
    "sp_ring": RingSequenceParallelStrategy,
}


@dataclass
class LowerAndPlanResult:
    """Phase-1 result: the byte-identical legacy result PLUS the module view."""

    apply_result: StrategyApplyResult  # the EXACT object apply_strategy_specs returns
    plans_by_role: dict[Any, list[AxisPlan]] = field(default_factory=dict)
    modules_by_role: dict[Any, StrategyPlan] = field(default_factory=dict)


def _build_module(spec: Any) -> StrategyModule:
    """Map one ``StrategySpec`` to its ``StrategyModule`` (Phase-1 stub forms)."""
    kind = spec.mesh_axis.kind
    size = spec.mesh_axis.size
    if kind == "stage_replica":
        routing_policy = getattr(spec.routing, "routing_policy", None)
        omni_lb_policy = _STAGE_POLICY_TO_OMNI_LB.get(routing_policy)
        return StageReplicaStrategy(size, omni_lb_policy)
    return _MODULE_BY_KIND[kind](size)


class Orchestrator:
    def lower_and_plan(
        self,
        stages: list[Any],
        strategy_specs: Mapping[Any, Sequence[Any]] | None,
    ) -> LowerAndPlanResult | None:
        """Phase-1 thin wrapper.

        Behavior contract: when ``strategy_specs`` is falsy → return ``None``
        (matches the current ``_apply_strategy_specs`` early-return). Otherwise:
          1. ``apply_result = apply_strategy_specs(stages, strategy_specs)`` — the
             UNCHANGED writer; the overlay + ``omni_lb_policy`` are identical by
             construction.
          2. For each role, build the ``StrategyModule`` list from the
             ``StrategySpec`` stack and call ``plan()`` on each (the module view).

        The module view is validated against ``apply_result.per_role_config`` by
        the §5.2 CPU equivalence test; in prod it is introspection only.
        """
        if not strategy_specs:
            return None
        apply_result = apply_strategy_specs(stages, strategy_specs)
        modules_by_role, plans_by_role = self._build_module_view(strategy_specs)
        return LowerAndPlanResult(apply_result, plans_by_role, modules_by_role)

    # --- module view (introspection / forward-compat); does NOT write stages ---
    def _build_module_view(
        self,
        strategy_specs: Mapping[Any, Sequence[Any]],
    ) -> tuple[dict[Any, StrategyPlan], dict[Any, list[AxisPlan]]]:
        modules_by_role: dict[Any, StrategyPlan] = {}
        plans_by_role: dict[Any, list[AxisPlan]] = {}
        for role, specs in strategy_specs.items():
            modules: StrategyPlan = []
            plans: list[AxisPlan] = []
            for spec in specs:
                kind = spec.mesh_axis.kind
                # Only translatable kinds reach here: apply_strategy_specs (above)
                # has already rejected reserved/unsupported kinds. Skip anything
                # without a Phase-1 module rather than fabricating a view.
                if kind != "stage_replica" and kind not in _MODULE_BY_KIND:
                    continue
                module = _build_module(spec)
                modules.append(module)
                plans.append(module.plan(LoweringCtx(spec=spec)))
            modules_by_role[role] = modules
            plans_by_role[role] = plans
        return modules_by_role, plans_by_role

    # --- GPU phases (skeleton in Phase 1; bodies land in Phase 2+) ---
    def build_groups(self, plan: StrategyPlan, ctx: Any) -> list:
        """Build per-axis process groups at worker init (Phase 2+)."""
        raise NotImplementedError("Orchestrator.build_groups lands in Phase 2+")

    def apply(self, plan: StrategyPlan, ctx: Any) -> list:
        """Apply per-axis model wiring at model init (Phase 2+)."""
        raise NotImplementedError("Orchestrator.apply lands in Phase 2+")

    # --- centralized validation (Phase 1: delegate to existing checks) ---
    def validate(self, plans: list[AxisPlan]) -> None:
        """Phase-1 no-op: validation is performed by ``apply_strategy_specs``
        (conflict-on-explicit, device-layout, EP==TP*DP, LB conflict) during
        ``lower_and_plan``. Centralized plan-level validation lands in Phase 2+.
        """
        return None
