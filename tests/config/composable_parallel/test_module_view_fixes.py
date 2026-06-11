# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the two deferred Phase-1 SHOULD-FIX items folded into 1b.

(a) Execution-type-aware ``tp`` / ``ep`` ownership in the module view: diffusion
    stages -> ``owned_by="omni"`` (+ ``rank_token``); AR / None stages ->
    ``owned_by="vllm"``. Proves ``LoweringCtx.execution_type`` is threaded.
(b) Fail-loud on an unmapped axis: a translatable kind with no ``StrategyModule``
    mapping raises :class:`UnmappedAxisError` instead of being silently dropped.
"""
from __future__ import annotations

import pytest

from vllm_omni.config.composable_parallel import (
    Broadcast,
    MeshAxisSpec,
    StrategySpec,
    TakeRank,
    UnmappedAxisError,
    Union,
)
from vllm_omni.config.composable_parallel.modules import axes as axes_mod
from vllm_omni.config.composable_parallel.modules import orchestrator as orch_mod
from vllm_omni.config.composable_parallel.modules.axes.ep import ExpertParallelStrategy
from vllm_omni.config.composable_parallel.modules.axes.tp import TensorParallelStrategy
from vllm_omni.config.composable_parallel.modules.base import LoweringCtx
from vllm_omni.config.composable_parallel.modules.orchestrator import Orchestrator
from vllm_omni.config.stage_config import StageConfig, StageExecutionType, StageType

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# --- spec / stage builders ---------------------------------------------------


def _tp(size: int) -> StrategySpec:
    return StrategySpec("tp", MeshAxisSpec("tp", size), Broadcast(), TakeRank())


def _ep(size: int) -> StrategySpec:
    return StrategySpec("ep", MeshAxisSpec("ep", size), Broadcast(), Union())


def _llm_stage(role: str = "thinker") -> StageConfig:
    return StageConfig(stage_id=0, model_stage=role, stage_type=StageType.LLM,
                       yaml_runtime={"num_replicas": 1})


def _diffusion_stage(role: str = "dit") -> StageConfig:
    return StageConfig(stage_id=0, model_stage=role, stage_type=StageType.DIFFUSION,
                       yaml_runtime={"num_replicas": 1})


# --- (a) execution-type-aware ownership: direct module-level ----------------


def test_tp_module_diffusion_is_omni_owned():
    plan = TensorParallelStrategy(2).plan(LoweringCtx(execution_type=StageExecutionType.DIFFUSION))
    assert plan.owned_by == "omni"
    assert plan.rank_token == "tp"
    assert plan.engine_kwargs == {"tensor_parallel_size": 2}


def test_tp_module_ar_is_vllm_owned():
    plan = TensorParallelStrategy(2).plan(LoweringCtx(execution_type=StageExecutionType.LLM_AR))
    assert plan.owned_by == "vllm"
    assert plan.engine_kwargs == {"tensor_parallel_size": 2}


def test_tp_module_none_execution_defaults_to_vllm():
    # No signal preserves pre-1b behavior.
    plan = TensorParallelStrategy(2).plan(LoweringCtx())
    assert plan.owned_by == "vllm"


def test_ep_module_diffusion_is_omni_owned():
    plan = ExpertParallelStrategy(2).plan(LoweringCtx(execution_type=StageExecutionType.DIFFUSION))
    assert plan.owned_by == "omni"
    assert plan.rank_token == "ep"
    assert plan.consumes_world_dim is False


def test_ep_module_ar_is_vllm_owned():
    plan = ExpertParallelStrategy(2).plan(LoweringCtx(execution_type=StageExecutionType.LLM_AR))
    assert plan.owned_by == "vllm"
    assert plan.consumes_world_dim is False


# --- (a) execution-type-aware ownership: through the orchestrator ------------


def test_orchestrator_threads_diffusion_execution_type():
    stages = [_diffusion_stage("dit")]
    result = Orchestrator().lower_and_plan(stages, {"dit": [_tp(2), _ep(2)]})
    assert result is not None
    by_axis = {p.axis: p for p in result.plans_by_role["dit"]}
    assert by_axis["tp"].owned_by == "omni"
    assert by_axis["tp"].rank_token == "tp"
    assert by_axis["ep"].owned_by == "omni"
    assert by_axis["ep"].rank_token == "ep"


def test_orchestrator_threads_ar_execution_type():
    stages = [_llm_stage("thinker")]
    result = Orchestrator().lower_and_plan(stages, {"thinker": [_tp(2), _ep(2)]})
    assert result is not None
    by_axis = {p.axis: p for p in result.plans_by_role["thinker"]}
    assert by_axis["tp"].owned_by == "vllm"
    assert by_axis["ep"].owned_by == "vllm"


# --- (b) fail-loud on an unmapped axis ---------------------------------------


def test_unmapped_axis_raises_loudly(monkeypatch):
    # Drop ``tp`` from the module mapping while it is STILL translatable, so
    # apply_strategy_specs succeeds but the module-view builder hits a kind it
    # cannot map -> must raise UnmappedAxisError (not silently drop it).
    patched = dict(orch_mod._MODULE_BY_KIND)
    patched.pop("tp")
    monkeypatch.setattr(orch_mod, "_MODULE_BY_KIND", patched)

    stages = [_llm_stage("thinker")]
    with pytest.raises(UnmappedAxisError) as exc:
        Orchestrator().lower_and_plan(stages, {"thinker": [_tp(2)]})
    assert "tp" in str(exc.value)
    assert "_MODULE_BY_KIND" in str(exc.value)


def test_unmapped_axis_message_mentions_role(monkeypatch):
    patched = dict(orch_mod._MODULE_BY_KIND)
    patched.pop("tp")
    monkeypatch.setattr(orch_mod, "_MODULE_BY_KIND", patched)
    stages = [_llm_stage("thinker")]
    with pytest.raises(UnmappedAxisError) as exc:
        Orchestrator().lower_and_plan(stages, {"thinker": [_tp(2)]})
    assert "thinker" in str(exc.value)


def test_axes_module_export_unchanged():
    # Guard that the axes package still exports the strategies the orchestrator
    # depends on (import-graph smoke check).
    assert hasattr(axes_mod, "TensorParallelStrategy")
    assert hasattr(axes_mod, "ExpertParallelStrategy")
