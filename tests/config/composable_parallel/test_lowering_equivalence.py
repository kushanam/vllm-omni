# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T2 lowering-equivalence matrix (§5.2): the core CPU safety net.

For a matrix of strategy stacks, assert the Phase-1 orchestrator path is
identical to today's ``apply_strategy_specs`` path on every observable:

* (a) the mutated stages' ``yaml_engine_args`` / ``yaml_runtime`` (deep-equal,
  which covers the nested ``parallel_config``);
* (b) ``omni_lb_policy``;
* (c) ``per_role_config`` / ``per_stage_config`` (``OmniParallelConfig`` equality);
* (d) raised exception type + message for the failure cases.

Because the orchestrator literally calls ``apply_strategy_specs`` for the writer,
(a)-(d) are equal by construction. The test ALSO asserts module-view
equivalence: for each role the aggregate of ``plan().engine_kwargs`` + degrees +
``owned_by`` over the built modules reproduces ``per_role_config[role]`` — the
lock that lets Phase 2 promote the module view to the writer.
"""
from __future__ import annotations

import pytest

from vllm_omni.config.composable_parallel import (
    AxisTranslationError,
    Broadcast,
    FanInByStage,
    GatherDim,
    MeshAxisSpec,
    PipelineMicrobatch,
    RouteByStage,
    ShardSequence,
    StitchPipeline,
    StrategyApplyError,
    StrategySpec,
    TakeRank,
    Union,
    apply_strategy_specs,
)
from vllm_omni.config.composable_parallel.modules.orchestrator import Orchestrator
from vllm_omni.config.stage_config import StageConfig, StageType

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# --- spec builders (mirror tests/config/composable_parallel/test_apply.py &
#     test_translator.py so the matrix constructs StrategySpec stacks identically)


def _tp(size: int) -> StrategySpec:
    return StrategySpec("tp", MeshAxisSpec("tp", size), Broadcast(), TakeRank())


def _dp(size: int, policy: str = "round_robin") -> StrategySpec:
    return StrategySpec("dp", MeshAxisSpec("dp", size), RouteByStage(policy), Union())


def _pp(size: int) -> StrategySpec:
    return StrategySpec("pp", MeshAxisSpec("pp", size), PipelineMicrobatch(), StitchPipeline())


def _ep(size: int) -> StrategySpec:
    return StrategySpec("ep", MeshAxisSpec("ep", size), Broadcast(), Union())


def _stage_replica(size: int, policy: str = "round_robin") -> StrategySpec:
    return StrategySpec(
        "stage_replica", MeshAxisSpec("stage_replica", size), RouteByStage(policy), FanInByStage()
    )


def _sp_ulysses(size: int) -> StrategySpec:
    return StrategySpec("sp_ulysses", MeshAxisSpec("sp_ulysses", size), ShardSequence(dim=1), GatherDim(dim=1))


def _sp_ring(size: int) -> StrategySpec:
    return StrategySpec("sp_ring", MeshAxisSpec("sp_ring", size), ShardSequence(dim=1), GatherDim(dim=1))


# --- stage builders (fresh each call: apply_strategy_specs mutates in place) ---


def _stage(stage_id: int, model_stage: str, engine_args=None, runtime=None) -> StageConfig:
    return StageConfig(
        stage_id=stage_id,
        model_stage=model_stage,
        yaml_engine_args=dict(engine_args or {}),
        yaml_runtime=dict(runtime or {"num_replicas": 1}),
    )


def _diffusion_stage(stage_id: int, model_stage: str, engine_args=None, runtime=None) -> StageConfig:
    return StageConfig(
        stage_id=stage_id,
        model_stage=model_stage,
        stage_type=StageType.DIFFUSION,
        yaml_engine_args=dict(engine_args or {}),
        yaml_runtime=dict(runtime or {"num_replicas": 1}),
    )


def _qwen_stages() -> list[StageConfig]:
    return [_stage(0, "thinker"), _stage(1, "talker"), _stage(2, "code2wav")]


def _qwen_stages_devices(role: str, devices: str) -> list[StageConfig]:
    stages = _qwen_stages()
    for s in stages:
        if s.model_stage == role:
            s.yaml_runtime["devices"] = devices
    return stages


def _dit_stage() -> list[StageConfig]:
    return [_diffusion_stage(0, "dit")]


def _dit_stage_devices(devices: str) -> list[StageConfig]:
    return [_diffusion_stage(0, "dit", runtime={"num_replicas": 1, "devices": devices})]


# --- equality helpers --------------------------------------------------------


def _assert_stages_equal(a: list[StageConfig], b: list[StageConfig]) -> None:
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert sa.stage_id == sb.stage_id
        # Deep-equal covers the nested ``parallel_config`` carried under
        # ``yaml_engine_args``.
        assert sa.yaml_engine_args == sb.yaml_engine_args
        assert sa.yaml_runtime == sb.yaml_runtime


def _module_engine_kwargs(plans) -> dict:
    """Aggregate the module plans into the as_engine_kwargs() shape.

    ``num_replicas`` / ``omni_lb_policy`` are surfaced on the stage_replica
    plan for the module view but are NOT per-stage engine args (the translator
    omits them from ``as_engine_kwargs``), so they are filtered out here.
    """
    merged: dict = {
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }
    for p in plans:
        for k, v in p.engine_kwargs.items():
            if k in ("num_replicas", "omni_lb_policy"):
                continue
            merged[k] = v
    return merged


# CONTRACT-defined ``owned_by`` per axis — WHO EXECUTES the parallelism
# (omni vs vLLM core), per docs/DESIGN_PARALLELISM_MODULARIZATION.md §5. This is
# ORTHOGONAL to the translator's ``l1_owner`` ("engine" vs "delegated", i.e. HOW
# routing is realized). Diffusion SP is the canonical example: l1_owner="engine"
# (it is applied as ulysses_degree/ring_degree engine kwargs) YET owned_by="omni"
# (omni's diffusion engine executes it; vLLM core does not). The two must NOT be
# equated — that conflation is exactly the defect this assertion guards against.
# ``tp`` / ``ep`` ownership is execution-type-aware (REVIEW_PHASE1_IMPL
# §SHOULD-FIX 1): omni-executed on diffusion stages, delegated to vLLM on AR
# stages. The other axes are execution-type-invariant.
_CONTRACT_OWNED_BY_AR = {
    "tp": "vllm",
    "dp": "vllm",
    "pp": "vllm",
    "ep": "vllm",
    "sp_ulysses": "omni",
    "sp_ring": "omni",
    "stage_replica": "omni",
    "vae_pp": "omni",
}
_CONTRACT_OWNED_BY_DIFFUSION = {**_CONTRACT_OWNED_BY_AR, "tp": "omni", "ep": "omni"}

# The translator's ``l1_owner`` is a SEPARATE, unchanged property of the
# translator output (not of the module ``owned_by``). Asserted independently.
_TRANSLATOR_L1_OWNER = {
    "tp": "engine",
    "dp": "engine",
    "pp": "engine",
    "ep": "engine",
    "sp_ulysses": "engine",
    "sp_ring": "engine",
    "stage_replica": "delegated",
}


def _diffusion_roles(stages: list[StageConfig]) -> set:
    """Roles whose stage is a diffusion stage (execution-type-aware ownership)."""
    return {s.model_stage for s in stages if StageType(s.stage_type) == StageType.DIFFUSION}


def _assert_module_view_matches(result, diffusion_roles: set) -> None:
    """Aggregate of module plans reproduces per_role_config[role] (§5.2).

    Two DISTINCT checks, deliberately kept separate:
      (1) module-view: the aggregate of ``plan().engine_kwargs`` + per-axis
          degrees reproduces the real translator output (``as_engine_kwargs`` +
          the ``OmniParallelConfig`` sizing fields), and each module's
          ``owned_by`` matches the CONTRACT-defined per-axis table (who
          executes: omni vs vllm).
      (2) translator ``l1_owner``: a property of the (unchanged) translator
          output ("engine" vs "delegated" routing). It is ORTHOGONAL to
          ``owned_by`` and is asserted on its own — never equated with it.
    """
    apply_result = result.apply_result
    for role, cfg in apply_result.per_role_config.items():
        plans = result.plans_by_role[role]
        by_axis = {p.axis: p for p in plans}
        owned_by_table = (
            _CONTRACT_OWNED_BY_DIFFUSION if role in diffusion_roles else _CONTRACT_OWNED_BY_AR
        )

        # (1) engine_kwargs aggregate == translator as_engine_kwargs.
        assert _module_engine_kwargs(plans) == dict(cfg.as_engine_kwargs())

        # (1b) per-axis degrees reproduce the translator-derived sizing.
        if "tp" in by_axis:
            assert by_axis["tp"].degree == cfg.tensor_parallel_size
        if "dp" in by_axis:
            assert by_axis["dp"].degree == cfg.data_parallel_size
        if "pp" in by_axis:
            assert by_axis["pp"].degree == cfg.pipeline_parallel_size
        if "ep" in by_axis:
            assert cfg.enable_expert_parallel is True
            # OmniParallelConfig stores no ep degree field; the translator
            # validates ep size == tp*dp, so that product is the authoritative
            # degree the ep module must reproduce.
            assert by_axis["ep"].degree == cfg.tensor_parallel_size * cfg.data_parallel_size
        if "sp_ulysses" in by_axis:
            assert by_axis["sp_ulysses"].degree == cfg.sp_ulysses_size
        if "sp_ring" in by_axis:
            assert by_axis["sp_ring"].degree == cfg.sp_ring_size
        if "stage_replica" in by_axis:
            sr = by_axis["stage_replica"]
            assert sr.degree == cfg.stage_replica_size
            assert sr.engine_kwargs.get("omni_lb_policy") == cfg.omni_lb_policy

        # (2a) owned_by matches the CONTRACT table (who executes) — NOT l1_owner.
        for axis_name, plan in by_axis.items():
            assert plan.owned_by == owned_by_table[axis_name]

        # (2b) SEPARATELY: the translator's l1_owner is its own unchanged
        # property ("engine"/"delegated"), orthogonal to the module owned_by.
        for axis_name in by_axis:
            assert cfg.l1_owners[axis_name] == _TRANSLATOR_L1_OWNER[axis_name]


# --- valid matrix ------------------------------------------------------------

# Each case: (id, make_stages, specs). ``make_stages`` is a fresh-stage factory.
_VALID_CASES = [
    ("tp", _qwen_stages, {"thinker": [_tp(2)]}),
    ("tp+dp", _qwen_stages, {"thinker": [_tp(2), _dp(2)]}),
    ("tp+pp", _qwen_stages, {"thinker": [_tp(2), _pp(2)]}),
    ("tp+dp+pp", _qwen_stages, {"thinker": [_tp(2), _dp(2), _pp(2)]}),
    ("ep_valid", _qwen_stages, {"thinker": [_tp(2), _ep(2)]}),
    ("stage_replica_random", _qwen_stages, {"talker": [_stage_replica(2, "random")]}),
    ("stage_replica_round_robin", _qwen_stages, {"talker": [_stage_replica(2, "round_robin")]}),
    ("stage_replica_least_queue", _qwen_stages, {"talker": [_stage_replica(2, "least_queue")]}),
    ("sp_ulysses", _dit_stage, {"dit": [_sp_ulysses(2)]}),
    ("sp_ring", _dit_stage, {"dit": [_sp_ring(2)]}),
    ("sp_ulysses+sp_ring", _dit_stage, {"dit": [_sp_ulysses(2), _sp_ring(2)]}),
    # SP degree==1: both the module plan and the translator must emit NO
    # ulysses_degree/ring_degree kwarg (emission is gated on degree>1). Verifies
    # the gating agrees on both paths.
    ("sp_ulysses_degree1", _dit_stage, {"dit": [_sp_ulysses(1)]}),
    ("sp_ring_degree1", _dit_stage, {"dit": [_sp_ring(1)]}),
    ("device_layout_pass", lambda: _qwen_stages_devices("thinker", "0,1"), {"thinker": [_tp(2)]}),
    ("sp_device_layout_pass", lambda: _dit_stage_devices("0,1,2,3"), {"dit": [_tp(2), _sp_ulysses(2)]}),
]


@pytest.mark.parametrize("case_id, make_stages, specs", _VALID_CASES, ids=[c[0] for c in _VALID_CASES])
def test_orchestrator_matches_baseline(case_id, make_stages, specs):
    # Baseline: today's path.
    base_stages = make_stages()
    baseline = apply_strategy_specs(base_stages, specs)

    # Orchestrator path (fresh stages, since apply mutates in place).
    orch_stages = make_stages()
    result = Orchestrator().lower_and_plan(orch_stages, specs)
    assert result is not None
    applied = result.apply_result

    # (a) mutated stages identical (deep-equal incl. nested parallel_config).
    _assert_stages_equal(base_stages, orch_stages)
    # (b) omni_lb_policy identical.
    assert applied.omni_lb_policy == baseline.omni_lb_policy
    # (c) per_role_config / per_stage_config identical (OmniParallelConfig eq).
    assert applied.per_role_config == baseline.per_role_config
    assert applied.per_stage_config == baseline.per_stage_config
    # module-view aggregate reproduces per_role_config.
    _assert_module_view_matches(result, _diffusion_roles(orch_stages))


# --- failure matrix: same exception type AND message from both paths ---------

_FAILURE_CASES = [
    ("ep_invalid", _qwen_stages, {"thinker": [_tp(2), _ep(4)]}, AxisTranslationError),
    (
        "conflicting_lb_policy",
        _qwen_stages,
        {
            "talker": [_stage_replica(2, "round_robin")],
            "code2wav": [_stage_replica(2, "least_queue")],
        },
        StrategyApplyError,
    ),
    (
        "device_layout_fail",
        lambda: _qwen_stages_devices("thinker", "0,1,2"),
        {"thinker": [_tp(2)]},
        StrategyApplyError,
    ),
    (
        "sp_device_layout_fail",
        lambda: _dit_stage_devices("0,1"),
        {"dit": [_tp(2), _sp_ulysses(2)]},
        StrategyApplyError,
    ),
]


@pytest.mark.parametrize(
    "case_id, make_stages, specs, exc_type", _FAILURE_CASES, ids=[c[0] for c in _FAILURE_CASES]
)
def test_orchestrator_matches_baseline_failures(case_id, make_stages, specs, exc_type):
    base_stages = make_stages()
    with pytest.raises(exc_type) as base_exc:
        apply_strategy_specs(base_stages, specs)

    orch_stages = make_stages()
    with pytest.raises(exc_type) as orch_exc:
        Orchestrator().lower_and_plan(orch_stages, specs)

    # Identical exception type AND message.
    assert type(orch_exc.value) is type(base_exc.value)
    assert str(orch_exc.value) == str(base_exc.value)


@pytest.mark.parametrize(
    "spec_fn, kwarg",
    [(_sp_ulysses, "ulysses_degree"), (_sp_ring, "ring_degree")],
    ids=["sp_ulysses", "sp_ring"],
)
def test_sp_degree_one_emits_no_kwarg_on_either_path(spec_fn, kwarg):
    # At degree==1 the kwarg is gated off: assert directly that NEITHER the
    # translator (as_engine_kwargs) NOR the module plan emits it.
    specs = {"dit": [spec_fn(1)]}
    base_stages = _dit_stage()
    baseline = apply_strategy_specs(base_stages, specs)
    orch_stages = _dit_stage()
    result = Orchestrator().lower_and_plan(orch_stages, specs)
    assert result is not None

    assert kwarg not in baseline.per_role_config["dit"].as_engine_kwargs()
    plan = result.plans_by_role["dit"][0]
    assert plan.degree == 1
    assert kwarg not in plan.engine_kwargs


def test_lower_and_plan_returns_none_without_specs():
    # Matches the _apply_strategy_specs early-return for the raw-deploy-arg
    # front-end (no strategy.yaml).
    stages = _qwen_stages()
    assert Orchestrator().lower_and_plan(stages, None) is None
    assert Orchestrator().lower_and_plan(stages, {}) is None
    # stages untouched.
    assert "tensor_parallel_size" not in stages[0].yaml_engine_args


# ---------------------------------------------------------------------------
# T8 (Phase 1c, §5.6 / §6.1.3 / §4.1.4) — init-dispatch subplan round-trip.
#
# For a representative strategy stack, the FILTERED config-time subplan
# ``[m for m in modules_by_role[role] if m.axis in INIT_DISPATCHABLE]``
# (with degree-active gating to mirror the runtime path) deep-equals the
# engine-time ``Orchestrator.lower_from_runtime_kwargs(od_config, exec_type)``
# output for the matching role.
#
# This is the lock that makes config-time and engine-time planners equivalent
# **for the axes Phase 1c actually dispatches** (§4.1.4). The comparison is
# against the **filtered InitDispatchPlan**, NOT against the full
# ``StrategyPlan`` (which still contains ``stage_replica`` / ``tp`` / ``ep``
# at config time and intentionally never crosses to the worker).
#
# Test cases are restricted to specs whose dispatchable axes have ``degree>1``
# — degree-1 init-dispatchable modules are emitted by the config-time module
# view but elided by the runtime path (which gates on ``degree>1``). The
# degree-active filter on the config side aligns the two paths so the
# deep-equal claim holds without additional normalization. The existing
# ``sp_*_degree1`` cases in the matrix above already exercise the
# config-time gating; T8 focuses on the round-trip.
# ---------------------------------------------------------------------------

# T8 round-trip imports — kept local to this section so the file's top-level
# import block stays append-only relative to its pre-T8 shape (T2 NIT
# follow-up). ``Orchestrator`` is already imported at the top because other
# tests in the matrix also use it; only the T8-specific symbols
# (``SimpleNamespace`` / ``LoweringCtx`` / ``INIT_DISPATCHABLE``) are
# colocated with the T8 round-trip code below.
from types import SimpleNamespace  # noqa: E402

from vllm_omni.config.composable_parallel.backends import (  # noqa: E402
    VLLM_BACKEND,
    effective_init_dispatch_axes,
)
from vllm_omni.config.composable_parallel.modules.base import LoweringCtx  # noqa: E402

# The effective init-dispatch set for the vLLM backend (replaces the deleted
# ``INIT_DISPATCHABLE`` constant): capability ∩ backend.delegated.
INIT_DISPATCHABLE = effective_init_dispatch_axes(VLLM_BACKEND)

# Hard local regression signal independent of production metadata: pin the
# literal expected set so a drift in the modules/backend table is caught here.
assert INIT_DISPATCHABLE == frozenset({"vae_pp", "sp_ulysses", "sp_ring"}), (
    f"unexpected init-dispatch set for vLLM backend: {sorted(INIT_DISPATCHABLE)}"
)


_T8_CASES = [
    ("sp_ulysses", _dit_stage, {"dit": [_sp_ulysses(2)]}),
    ("sp_ring", _dit_stage, {"dit": [_sp_ring(2)]}),
    ("sp_ulysses+sp_ring", _dit_stage, {"dit": [_sp_ulysses(2), _sp_ring(2)]}),
    # Degree-1 / no-op case: both paths produce an empty subplan (a non-trivial
    # equivalence claim — the config-time side filters by INIT_DISPATCHABLE and
    # degree, the engine-time side filters by degree at construction).
    ("no_dispatchable_axes", _dit_stage, {"dit": [_tp(2)]}),
]


def _filtered_config_time_subplan(modules: list):
    """The filtered InitDispatchPlan from a config-time module list (§4.1.4).

    Includes modules whose axis is in :data:`INIT_DISPATCHABLE` AND whose
    plan() reports an active (``degree>1``) axis — matching the engine-time
    path's gating so the round-trip deep-equal holds.
    """
    out = []
    for m in modules:
        if m.axis not in INIT_DISPATCHABLE:
            continue
        if m.plan(LoweringCtx()).degree <= 1:
            continue
        out.append(m)
    return out


def _od_config_from(per_role_cfg) -> SimpleNamespace:
    """Build a minimal od_config stand-in carrying the three runtime degrees.

    Mirrors what ``OmniDiffusionConfig.parallel_config`` would expose for the
    dispatchable axes; non-dispatchable fields are set to sentinel-active
    values (``tensor_parallel_size`` etc.) to lock down the F1 / §5.2 O1
    invariant that the runtime path ignores them.
    """
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            # vae_pp cannot be expressed via strategy specs today (it is a
            # reserved kind the translator rejects), so the round-trip starts
            # from vae_patch_parallel_size=1 in T8.
            vae_patch_parallel_size=1,
            ulysses_degree=per_role_cfg.sp_ulysses_size,
            ring_degree=per_role_cfg.sp_ring_size,
            tensor_parallel_size=per_role_cfg.tensor_parallel_size,
            data_parallel_size=per_role_cfg.data_parallel_size,
            pipeline_parallel_size=per_role_cfg.pipeline_parallel_size,
        ),
    )


@pytest.mark.parametrize(
    "case_id, make_stages, specs", _T8_CASES, ids=[c[0] for c in _T8_CASES]
)
def test_init_dispatch_subplan_round_trip(case_id, make_stages, specs):
    stages = make_stages()
    result = Orchestrator().lower_and_plan(stages, specs)
    assert result is not None

    for role, modules in result.modules_by_role.items():
        expected = _filtered_config_time_subplan(modules)
        per_role_cfg = result.apply_result.per_role_config[role]
        od_config = _od_config_from(per_role_cfg)

        actual = Orchestrator().lower_from_runtime_kwargs(
            od_config, execution_type=None
        )

        # Deep-equal on AxisPlan: covers axes, degrees, owned_by,
        # engine_kwargs, rank_token, consumes_world_dim (§4.1.4).
        expected_plans = [m.plan(LoweringCtx()) for m in expected]
        actual_plans = [m.plan(LoweringCtx()) for m in actual]
        assert expected_plans == actual_plans, (
            f"role={role!r} subplan mismatch: "
            f"expected={expected_plans} actual={actual_plans}"
        )
