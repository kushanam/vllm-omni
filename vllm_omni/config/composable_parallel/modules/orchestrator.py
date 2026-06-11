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
    _resolve_stage,
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
from vllm_omni.config.composable_parallel.translator import (
    _STAGE_POLICY_TO_OMNI_LB,
    UnmappedAxisError,
)

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


def _stage_execution_type(stage: Any) -> object | None:
    """Resolve a stage's execution type for the module-view ownership decision.

    Prefers an explicit ``execution_type`` (newer resolved stage configs); else
    derives it from the legacy ``stage_type`` (DIFFUSION -> diffusion execution,
    everything else -> AR). ``None`` when the stage carries no signal. Imported
    lazily to keep this module's import graph light and cycle-free.
    """
    et = getattr(stage, "execution_type", None)
    if et is not None:
        return et
    from vllm_omni.config.stage_config import StageExecutionType, StageType

    if getattr(stage, "stage_type", None) == StageType.DIFFUSION:
        return StageExecutionType.DIFFUSION
    return StageExecutionType.LLM_AR


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
        modules_by_role, plans_by_role = self._build_module_view(stages, strategy_specs)
        return LowerAndPlanResult(apply_result, plans_by_role, modules_by_role)

    # --- module view (introspection / forward-compat); does NOT write stages ---
    def _build_module_view(
        self,
        stages: list[Any],
        strategy_specs: Mapping[Any, Sequence[Any]],
    ) -> tuple[dict[Any, StrategyPlan], dict[Any, list[AxisPlan]]]:
        modules_by_role: dict[Any, StrategyPlan] = {}
        plans_by_role: dict[Any, list[AxisPlan]] = {}
        for role, specs in strategy_specs.items():
            # Resolve the role's stage execution type so execution-type-sensitive
            # axes (tp/ep) report faithful ownership (omni vs vLLM). Role->stage
            # matching already succeeded in apply_strategy_specs above.
            execution_type = _stage_execution_type(_resolve_stage(stages, role))
            modules: StrategyPlan = []
            plans: list[AxisPlan] = []
            for spec in specs:
                kind = spec.mesh_axis.kind
                # apply_strategy_specs (above) already rejected reserved /
                # unsupported kinds, so anything reaching here is translatable.
                # A translatable kind with no module mapping is a developer error
                # (a supported axis silently vanishing from the view), so FAIL
                # LOUDLY rather than dropping it (REVIEW_PHASE1_IMPL §SHOULD-FIX 2).
                if kind != "stage_replica" and kind not in _MODULE_BY_KIND:
                    raise UnmappedAxisError(
                        f"role {role!r}: axis kind {kind!r} was accepted by the "
                        f"translator but has no StrategyModule mapping in "
                        f"_MODULE_BY_KIND. Add a module for it or remove it from "
                        f"the translator's supported kinds. (Phase 1b fail-loud, "
                        f"REVIEW_PHASE1_IMPL §SHOULD-FIX 2.)"
                    )
                module = _build_module(spec)
                modules.append(module)
                plans.append(module.plan(LoweringCtx(spec=spec, execution_type=execution_type)))
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
