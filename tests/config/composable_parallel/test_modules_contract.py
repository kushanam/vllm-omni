# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T1: construction/validation of the StrategyModule contract types (§5.1)."""

from __future__ import annotations

import ast
import inspect

import pytest

from vllm_omni.config.composable_parallel.modules.base import (
    ApplyCtx,
    AxisPlan,
    AxisResult,
    DelegatedStrategy,
    GroupBuildCtx,
    GroupHandle,
    LoweringCtx,
    OmniExecutedStrategy,
    StrategyModule,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_axis_plan_minimal_defaults():
    plan = AxisPlan(axis="tp", degree=4, owned_by="vllm")
    assert plan.axis == "tp"
    assert plan.degree == 4
    assert plan.owned_by == "vllm"
    assert plan.engine_kwargs == {}
    assert plan.rank_token is None
    assert plan.consumes_world_dim is True
    assert plan.requires == frozenset()


def test_axis_plan_full_fields():
    plan = AxisPlan(
        axis="sp_ulysses",
        degree=2,
        owned_by="omni",
        engine_kwargs={"ulysses_degree": 2},
        rank_token="sp",
        consumes_world_dim=True,
        requires=frozenset({"sp_descriptor"}),
    )
    assert plan.engine_kwargs == {"ulysses_degree": 2}
    assert plan.rank_token == "sp"
    assert plan.requires == frozenset({"sp_descriptor"})


def test_group_handle_minimal_defaults():
    handle = GroupHandle(axis="tp", parallel_mode="tensor")
    assert handle.axis == "tp"
    assert handle.parallel_mode == "tensor"
    assert handle.coordinator is None
    assert handle.ranks == ()
    assert handle.delegated is False
    assert handle.reused is False


def test_group_handle_full_fields():
    handle = GroupHandle(
        axis="vae_pp",
        parallel_mode="reused",
        coordinator=None,
        ranks=((0, 1), (2, 3)),
        delegated=False,
        reused=True,
    )
    assert handle.ranks == ((0, 1), (2, 3))
    assert handle.reused is True


def test_axis_result_minimal_defaults():
    result = AxisResult(axis="tp")
    assert result.axis == "tp"
    assert result.group is None
    assert result.hooks_applied == 0
    assert result.notes == ()


def test_axis_result_delegated_classmethod():
    result = AxisResult.delegated("tp")
    assert result.axis == "tp"
    assert result.group is not None
    assert result.group.delegated is True
    assert result.group.parallel_mode == "delegated"
    assert result.group.coordinator is None
    assert result.group.axis == "tp"


def test_delegated_strategy_build_and_apply_are_typed_noops():
    class _Delegate(DelegatedStrategy):
        axis = "tp"

    delegate = _Delegate()
    build_result = delegate.build_groups(
        GroupBuildCtx(rank_generator=object(), backend="nccl", world_size=4)
    )
    apply_result = delegate.apply(ApplyCtx(model=object(), od_config=object()))

    for result in (build_result, apply_result):
        assert isinstance(result, AxisResult)
        assert result.axis == "tp"
        assert result.group is not None
        assert result.group.delegated is True
        assert result.group.parallel_mode == "delegated"


def test_base_has_no_top_level_torch_import():
    """Deterministic static check: base.py must be torch-free at module top level.

    torch / ProcessGroup / GroupCoordinator may only be imported inside the
    ``if TYPE_CHECKING:`` block, never as a direct child of the module body.
    Parsing the source with ``ast`` makes this robust to whatever else the test
    suite has already imported into ``sys.modules``.
    """
    import vllm_omni.config.composable_parallel.modules.base as base_mod

    forbidden = ("torch", "vllm_omni.diffusion.distributed.parallel_state")

    def _is_forbidden(module_name: str | None) -> bool:
        if module_name is None:
            return False
        return any(
            module_name == f or module_name.startswith(f + ".") for f in forbidden
        )

    source = inspect.getsource(base_mod)
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not _is_forbidden(alias.name), (
                    f"top-level torch import: {alias.name!r}"
                )
        elif isinstance(node, ast.ImportFrom):
            assert not _is_forbidden(node.module), (
                f"top-level torch import-from: {node.module!r}"
            )


def test_runtime_checkable_isinstance_for_bases():
    class _Omni(OmniExecutedStrategy):
        axis = "sp_ulysses"

    class _Delegate(DelegatedStrategy):
        axis = "tp"

    assert isinstance(_Omni(), StrategyModule)
    assert isinstance(_Delegate(), StrategyModule)


def test_lowering_ctx_defaults():
    ctx = LoweringCtx()
    assert ctx.spec is None
    assert ctx.raw_degree is None
    assert ctx.execution_type is None
    assert ctx.shard_extension == {}
