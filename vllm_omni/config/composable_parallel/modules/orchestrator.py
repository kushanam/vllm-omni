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
from vllm_omni.config.composable_parallel.backends import (
    BackendAxisOwnership,
    VLLM_BACKEND,
)
from vllm_omni.config.composable_parallel.modules.axes import (
    DataParallelStrategy,
    ExpertParallelStrategy,
    PipelineParallelStrategy,
    RingSequenceParallelStrategy,
    StageReplicaStrategy,
    TensorParallelStrategy,
    UlyssesSequenceParallelStrategy,
    VaePatchParallelStrategy,
)
from vllm_omni.config.composable_parallel.modules.base import (
    ApplyCtx,
    AxisName,
    AxisPlan,
    AxisResult,
    LoweringCtx,
    StrategyModule,
)
from vllm_omni.config.composable_parallel.translator import (
    _STAGE_POLICY_TO_OMNI_LB,
    UnmappedAxisError,
)

# The FULL logical plan built at config time (includes stage_replica / tp / ep,
# anything the translator accepts). Returned by ``lower_and_plan`` for
# introspection. NOT what crosses the config→engine boundary at runtime.
StrategyPlan = list[StrategyModule]

# The init-dispatchable subplan rebuilt at engine init by
# ``Orchestrator.lower_from_runtime_kwargs``. A flat list of ``StrategyModule``
# instances that are init-dispatched for the active backend — i.e. each module
# satisfies ``module.supports_init_dispatch and module.axis in
# backend.delegated`` (the intersection rule, design §2.3). Intentionally
# excludes ``stage_replica`` / ``tp`` / ``dp`` / ``pp`` / ``ep`` (native to vLLM)
# so the runtime dispatcher only sees axes whose ``apply()`` actually runs at
# init (Phase 1c §4.1.2 / §4.1.4).
InitDispatchPlan = list[StrategyModule]

# ---------------------------------------------------------------------------
# Init-time dispatch contract (Phase 1c §4.1.2 / §4.4; modularization design §2)
# ---------------------------------------------------------------------------
# There is NO global init-dispatch allowlist any more. "Is this module
# init-dispatchable?" is answered by the intersection of two orthogonal,
# co-located declarations:
#   * the per-module capability ``StrategyModule.supports_init_dispatch``
#     (base.py / each axis module), and
#   * the per-backend ``BackendAxisOwnership.delegated`` set (backends.py).
# Apply-time ordering likewise moved off the central ``APPLY_ORDER`` tuple onto
# each module's ``init_dispatch_order`` key; :meth:`Orchestrator.apply` sorts
# dispatchable modules by ``(init_dispatch_order, axis)``. For the vLLM backend
# this reproduces today's exact set ``{vae_pp, sp_ulysses, sp_ring}`` and order
# ``vae_pp -> sp_ring -> sp_ulysses`` (design §3).


# ---------------------------------------------------------------------------
# Orchestrator-level exception types (§4.1.5)
# ---------------------------------------------------------------------------
class OrchestratorError(RuntimeError):
    """Raised by :class:`Orchestrator` for plan-shape errors at the dispatch
    boundary (e.g. duplicate axes in an :class:`InitDispatchPlan`). Distinct
    from :class:`UnmappedAxisError` (which is raised for axes that have no
    init-time dispatch slot at all) so callers can target each cause precisely.
    """


class RuntimePlanReconstructionError(OrchestratorError):
    """Raised by :meth:`Orchestrator.lower_from_runtime_kwargs` when the
    init-dispatchable subplan cannot be rebuilt from the resolved
    :class:`DiffusionParallelConfig` (§4.1.5 case 3).

    Carries the offending ``od_config`` for diagnostic purposes; the project
    lead's user-facing surface is a hard fail at engine init, never a silent
    inline fallback (anti-pattern explicitly rejected by §4.1.5).
    """

    def __init__(self, message: str, od_config: object) -> None:
        super().__init__(message)
        self.od_config = od_config


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
    def __init__(self, backend: BackendAxisOwnership = VLLM_BACKEND) -> None:
        """The orchestrator is a pure function of its inputs plus the active
        backend's axis-ownership table (design §2.2, open question 2). ``backend``
        defaults to :data:`VLLM_BACKEND` — the only backend today — so existing
        ``Orchestrator()`` call sites are unaffected. A non-vLLM backend passes
        its own :class:`BackendAxisOwnership` here with no other code change.
        """
        self.backend = backend

    def _is_init_dispatchable(self, module: StrategyModule) -> bool:
        """The intersection rule (design §2.3): a module is init-dispatched iff
        it declares the capability AND the active backend delegates its axis."""
        return bool(
            getattr(module, "supports_init_dispatch", False)
            and module.axis in self.backend.delegated
        )

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
                # Thread the active backend so execution-type-sensitive modules
                # (tp/ep) — and every module, for single-source-of-truth — resolve
                # ``owned_by`` from ``self.backend.executes`` via
                # ``axis_execution_owner`` (design §3.3). Default None would resolve
                # to VLLM_BACKEND anyway; passing it explicitly keeps a non-vLLM
                # backend's ownership faithful with no other change.
                plans.append(
                    module.plan(
                        LoweringCtx(
                            spec=spec,
                            execution_type=execution_type,
                            backend=self.backend,
                        )
                    )
                )
            modules_by_role[role] = modules
            plans_by_role[role] = plans
        return modules_by_role, plans_by_role

    # --- GPU phases (skeleton in Phase 1; bodies land in Phase 2+) ---
    def build_groups(self, plan: StrategyPlan, ctx: Any) -> list:
        """Build per-axis process groups at worker init (Phase 2+)."""
        raise NotImplementedError("Orchestrator.build_groups lands in Phase 2+")

    def apply(self, plan: InitDispatchPlan, ctx: ApplyCtx) -> list[AxisResult]:
        """Init-time dispatch loop (Phase 1c §4.2; modularization design §2.3).

        Dispatches every module that is init-dispatchable for the active backend
        (``module.supports_init_dispatch and module.axis in
        self.backend.delegated``), calling ``module.apply(ctx)`` and collecting
        its :class:`AxisResult`. Modules run sorted by
        ``(init_dispatch_order, axis)``, so the returned list is in execution
        order (NOT input order). For the vLLM backend this reproduces today's
        ``vae_pp -> sp_ring -> sp_ulysses`` ordering exactly.

        Failure modes (fail-loud, no global ``try/except``):
          * ``plan`` contains a module that is NOT init-dispatchable for the
            backend (capability flag off, or axis not delegated) →
            :class:`UnmappedAxisError` (the dispatcher honestly cannot apply it;
            see §4.1.5 case 1).
          * ``plan`` contains the same axis twice →
            :class:`OrchestratorError` (developer error; the runtime
            reconstruction path never produces duplicates).
          * a module's ``apply()`` raises → exception propagates verbatim
            (the orchestrator adds no try/except; SP-specific warn-and-continue
            parity lives inside each SP module's own wrapper per §4.5.4 row 6
            / §5.5 A4 / N1 deferred).
        """
        # 1. Fail loud on any module we are not equipped to dispatch — i.e. one
        # whose axis is not delegated by the active backend, or whose module
        # does not declare the init-dispatch capability (the two-gate rule).
        for module in plan:
            if not self._is_init_dispatchable(module):
                axis = module.axis
                raise UnmappedAxisError(
                    f"Orchestrator.apply received a module for axis {axis!r} "
                    "that is NOT init-dispatchable for backend "
                    f"{self.backend.name!r}: it requires "
                    "module.supports_init_dispatch is True "
                    f"(got {getattr(module, 'supports_init_dispatch', False)!r}) "
                    "AND the axis in the backend's delegated set "
                    f"({sorted(self.backend.delegated)}). Promote the axis by "
                    "setting supports_init_dispatch=True on its module (next to a "
                    "real apply()) and adding it to the backend's delegated set, "
                    "or exclude it from the InitDispatchPlan."
                )

        # 2. Build axis -> module map; reject duplicates loudly (the runtime
        # reconstruction never produces duplicates; this catches a developer
        # error if a hand-built plan is ever passed in).
        by_axis: dict[AxisName, StrategyModule] = {}
        for module in plan:
            if module.axis in by_axis:
                raise OrchestratorError(
                    f"duplicate axis {module.axis!r} in InitDispatchPlan; "
                    f"each axis must appear at most once."
                )
            by_axis[module.axis] = module

        # 3. Dispatch in ascending (init_dispatch_order, axis) order. Sorting on
        # the per-module key (not a central tuple) is what dissolves APPLY_ORDER;
        # the axis tiebreak keeps the order stable/deterministic on accidental
        # ties (design §6 "order semantics" risk).
        results: list[AxisResult] = []
        for module in sorted(
            by_axis.values(), key=lambda m: (m.init_dispatch_order, m.axis)
        ):
            # Failures here propagate (see docstring); the dispatch loop is a
            # thin iterator with no global exception sink.
            results.append(module.apply(ctx))
        return results

    def lower_from_runtime_kwargs(
        self,
        od_config: Any,
        execution_type: object | None,
    ) -> InitDispatchPlan:
        """Rebuild the init-dispatchable subplan from the resolved config (§4.1.2).

        Pure function of ``(od_config.parallel_config, execution_type)`` —
        nothing module-level crosses the config→engine boundary. The result is
        a list of :class:`StrategyModule` instances that are init-dispatchable
        for the active backend (``supports_init_dispatch and axis in
        backend.delegated``) and currently active (``degree>1`` on their
        corresponding ``DiffusionParallelConfig`` field).

        Critically: this function NEVER returns a module for an axis the backend
        does not delegate. Even when ``od_config.parallel_config`` implies
        ``stage_replica`` / ``tp`` / ``dp`` / ``pp`` / ``ep`` axes, those are
        intentionally absent from the returned plan — they live only on the
        config-time :class:`StrategyPlan` (§4.1.2, §5.2 O1).

        ``execution_type`` is accepted for API symmetry with future axes whose
        ``plan()`` is execution-type-sensitive. Today's three dispatchable
        axes (``vae_pp`` / ``sp_ulysses`` / ``sp_ring``) are all
        execution-type-invariant, so the value is unused here.

        Failure mode (§4.1.5 case 3): any failure parsing
        ``od_config.parallel_config`` (missing attribute, non-numeric degree,
        etc.) is wrapped in :class:`RuntimePlanReconstructionError`. No silent
        fallback to the inline path.
        """
        del execution_type  # forward-compat only; see docstring.
        try:
            parallel_config = od_config.parallel_config
            candidates: list[StrategyModule] = []
            # vae_pp: active iff vae_patch_parallel_size > 1.
            vae_pp_size = int(parallel_config.vae_patch_parallel_size)
            if vae_pp_size > 1:
                candidates.append(VaePatchParallelStrategy(vae_pp_size))
            # sp_ulysses: active iff ulysses_degree > 1.
            ulysses_degree = int(parallel_config.ulysses_degree)
            if ulysses_degree > 1:
                candidates.append(UlyssesSequenceParallelStrategy(ulysses_degree))
            # sp_ring: active iff ring_degree > 1.
            ring_degree = int(parallel_config.ring_degree)
            if ring_degree > 1:
                candidates.append(RingSequenceParallelStrategy(ring_degree))
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimePlanReconstructionError(
                "Failed to reconstruct InitDispatchPlan from "
                f"od_config.parallel_config: {exc}",
                od_config,
            ) from exc

        # Mandatory belt-and-braces filter (§5.2 O1): if a future edit adds a
        # candidate module that is not init-dispatchable for the active backend
        # (capability flag off, or axis not delegated), drop it here rather than
        # letting it cross the dispatch boundary.
        return [m for m in candidates if self._is_init_dispatchable(m)]

    # --- centralized validation (Phase 1: delegate to existing checks) ---
    def validate(self, plans: list[AxisPlan]) -> None:
        """Phase-1 no-op: validation is performed by ``apply_strategy_specs``
        (conflict-on-explicit, device-layout, EP==TP*DP, LB conflict) during
        ``lower_and_plan``. Centralized plan-level validation lands in Phase 2+.
        """
        return None
