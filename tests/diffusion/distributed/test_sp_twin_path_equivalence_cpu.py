# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 1c-Twin §7.1 — CPU equivalence test for the twin SP runtime paths.

For the same ``(_FakePipeline, DiffusionParallelConfig)`` input, the LEGACY
path (``hooks.sequence_parallel._apply_sp_runtime``) and the NEW path
(``distributed.sp_runtime.apply_sp_to_pipeline``) MUST:

1. Make identical calls to the bottom installers
   (``apply_sequence_parallel`` / ``apply_sequence_parallel_from_descriptor``)
   in the same order, with structurally equal arguments.
2. Write the same value to ``ForwardContext.sp_plan_hooks_applied``.
3. Return the same ``applied_count``.

CRITICAL — TWIN-1 mock-target rule. ``sp_runtime.py`` imports the two
installers at module import time with ``from ... import ...``; the resulting
*local* names live inside ``vllm_omni.diffusion.distributed.sp_runtime``.
Patching the legacy module path does NOT redirect the new path's local
names (classic Python "patch where it's looked up, not where it's
defined"). The patches below therefore target BOTH module namespaces — the
legacy one for ``_apply_sp_runtime`` and the new one for
``apply_sp_to_pipeline`` — so a single test run can drive both paths and
compare their installer call sequences side-by-side.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
import torch.nn as nn

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_runtime import apply_sp_to_pipeline
from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    override_forward_context,
)
from vllm_omni.diffusion.hooks.sequence_parallel import _apply_sp_runtime

pytestmark = [pytest.mark.cpu]


# Mock-target string constants. The full-module-path strings are part of
# the contract (TWIN-1 lock-in): patching the legacy installer names alone
# would silently miss every NEW-path installer call.
_NEW_PATH_INSTALLER = (
    "vllm_omni.diffusion.distributed.sp_runtime.apply_sequence_parallel"
)
_NEW_PATH_DESCRIPTOR_INSTALLER = (
    "vllm_omni.diffusion.distributed.sp_runtime.apply_sequence_parallel_from_descriptor"
)
_LEGACY_PATH_INSTALLER = (
    "vllm_omni.diffusion.hooks.sequence_parallel.apply_sequence_parallel"
)
_LEGACY_PATH_DESCRIPTOR_INSTALLER = (
    "vllm_omni.diffusion.hooks.sequence_parallel.apply_sequence_parallel_from_descriptor"
)
_GET_SP_PLAN = "vllm_omni.diffusion.distributed.sp_plan.get_sp_plan_from_model"


# -----------------------------------------------------------------------------
# Fakes
# -----------------------------------------------------------------------------


class _FakeTransformer(nn.Module):
    """Stand-in for a real DiT/transformer module.

    ``with_descriptor=True`` attaches an ``_sp_descriptor`` marker so the
    descriptor branch can be exercised without depending on a real
    ``SPDescriptor`` subclass. Tests mock both installer functions, so the
    descriptor's actual type never matters at runtime.
    """

    def __init__(self, *, with_descriptor: bool = False) -> None:
        super().__init__()
        if with_descriptor:
            self._sp_descriptor = object()


class _FakePipeline(nn.Module):
    """Single-transformer pipeline whose class name maps to
    :class:`DefaultAdapter` (no override registered in
    ``_PIPELINE_TRANSFORMER_ADAPTERS``)."""

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer


class _FakeWan22Pipeline(nn.Module):
    """Multi-transformer fake mirroring the live shape at
    ``vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py:377,:383``.

    The new path uses :class:`Wan22Adapter` only when the pipeline class is
    registered to it; tests use ``monkeypatch.setitem`` to register
    ``_FakeWan22Pipeline`` under :class:`Wan22Adapter` without polluting
    the real registry.
    """

    def __init__(self, t1: nn.Module, t2: nn.Module) -> None:
        super().__init__()
        self.transformer = t1
        self.transformer_2 = t2


class _NoTransformerPipeline(nn.Module):
    """Pipeline with no ``transformer`` attribute at all — drives the
    new-path adapter-specific warning (TWIN-3 Option A) and the legacy
    "no SP plan/descriptor" warning. The class name is not registered in
    ``_PIPELINE_TRANSFORMER_ADAPTERS`` so :class:`DefaultAdapter` returns
    an empty list."""

    def __init__(self) -> None:
        super().__init__()


def _make_parallel_config(
    *,
    ulysses_degree: int = 2,
    ring_degree: int = 1,
    use_sp_descriptor: bool = False,
    use_init_dispatch: bool = True,
) -> DiffusionParallelConfig:
    return DiffusionParallelConfig(
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        use_sp_descriptor=use_sp_descriptor,
        use_init_dispatch=use_init_dispatch,
    )


def _make_forward_context(
    parallel_config: DiffusionParallelConfig,
) -> ForwardContext:
    od = OmniDiffusionConfig()
    od.parallel_config = parallel_config
    return ForwardContext(omni_diffusion_config=od)


def _normalize(c: call) -> call:
    """Drop the bound transformer module (positional 0) since the two
    runs see two different module instances; keep everything else via
    ``unittest.mock.call(...)`` equality."""
    args, kwargs = c.args, c.kwargs
    return call(*args[1:], **kwargs)


# -----------------------------------------------------------------------------
# Test 1: equivalence on a single-transformer (DefaultAdapter) pipeline.
# -----------------------------------------------------------------------------


def test_twin_path_equivalence_on_single_transformer_pipeline():
    """Smoke test: legacy ``_apply_sp_runtime`` and new
    ``apply_sp_to_pipeline`` produce identical observable behavior on a
    single-transformer pipeline (DefaultAdapter shape).

    Asserts identical installer ``call_args_list`` (modulo per-run
    transformer instance identity), identical ``applied_count``, and
    identical ``ForwardContext.sp_plan_hooks_applied`` final state.
    """
    parallel_config = _make_parallel_config(
        ulysses_degree=2, ring_degree=1, use_sp_descriptor=False,
    )

    # Both paths must resolve the SAME plan object so the installer receives
    # an identical plan argument; MagicMock compares by identity, so two
    # separate instances would break the call_args_list equality below.
    shared_plan = MagicMock(name="fake_plan")

    # ---- Legacy path ----
    model_old = _FakePipeline(_FakeTransformer())
    fc_old = _make_forward_context(parallel_config)
    with (
        override_forward_context(fc_old),
        patch(_LEGACY_PATH_INSTALLER) as old_legacy_inst,
        patch(_LEGACY_PATH_DESCRIPTOR_INSTALLER, return_value=True) as old_desc_inst,
        patch(_GET_SP_PLAN, return_value=shared_plan),
    ):
        old_count = _apply_sp_runtime(model_old, parallel_config)
        old_legacy_calls = list(old_legacy_inst.call_args_list)
        old_desc_calls = list(old_desc_inst.call_args_list)
        old_marker = fc_old.sp_plan_hooks_applied

    # ---- New path ----
    model_new = _FakePipeline(_FakeTransformer())
    fc_new = _make_forward_context(parallel_config)
    with (
        override_forward_context(fc_new),
        patch(_NEW_PATH_INSTALLER) as new_legacy_inst,
        patch(_NEW_PATH_DESCRIPTOR_INSTALLER, return_value=True) as new_desc_inst,
        patch(_GET_SP_PLAN, return_value=shared_plan),
    ):
        new_count = apply_sp_to_pipeline(model_new, parallel_config)
        new_legacy_calls = list(new_legacy_inst.call_args_list)
        new_desc_calls = list(new_desc_inst.call_args_list)
        new_marker = fc_new.sp_plan_hooks_applied

    assert old_count == new_count, (
        f"applied_count mismatch: old={old_count} new={new_count}"
    )
    assert old_marker == new_marker, (
        f"sp_plan_hooks_applied mismatch: old={old_marker} new={new_marker}"
    )
    assert len(old_legacy_calls) == len(new_legacy_calls)
    assert len(old_desc_calls) == len(new_desc_calls)
    assert [_normalize(c) for c in old_legacy_calls] == [
        _normalize(c) for c in new_legacy_calls
    ]
    assert [_normalize(c) for c in old_desc_calls] == [
        _normalize(c) for c in new_desc_calls
    ]


# -----------------------------------------------------------------------------
# Test 2: equivalence + ordering on a Wan2.2-style two-transformer pipeline.
# -----------------------------------------------------------------------------


def test_twin_path_equivalence_on_two_transformer_pipeline(monkeypatch):
    """Wan2.2-shaped fake exercises adapter ordering and two-target
    selection. Both paths MUST produce installer calls in the SAME order
    on the SAME ``(transformer, transformer_2)`` slot pair.

    Legacy reaches the two transformers via the hardcoded
    ``transformer_attrs = ["transformer", "transformer_2", ...]`` scan; the
    new path reaches them via :class:`Wan22Adapter`. The class name
    ``_FakeWan22Pipeline`` is registered to :class:`Wan22Adapter` for the
    duration of this test only via ``monkeypatch.setitem``.
    """
    from vllm_omni.diffusion.distributed.pipeline_adapters import (
        _PIPELINE_TRANSFORMER_ADAPTERS,
        Wan22Adapter,
    )

    monkeypatch.setitem(
        _PIPELINE_TRANSFORMER_ADAPTERS,
        "_FakeWan22Pipeline",
        Wan22Adapter,
    )

    parallel_config = _make_parallel_config(
        ulysses_degree=2, ring_degree=1, use_sp_descriptor=False,
    )

    def _build_pipeline() -> _FakeWan22Pipeline:
        return _FakeWan22Pipeline(_FakeTransformer(), _FakeTransformer())

    # Both paths must resolve the SAME plan object so the installer receives
    # an identical plan argument; MagicMock compares by identity, so two
    # separate instances would break the call_args_list equality below.
    shared_plan = MagicMock(name="fake_plan")

    # ---- Legacy path ----
    model_old = _build_pipeline()
    fc_old = _make_forward_context(parallel_config)
    with (
        override_forward_context(fc_old),
        patch(_LEGACY_PATH_INSTALLER) as old_legacy_inst,
        patch(_LEGACY_PATH_DESCRIPTOR_INSTALLER, return_value=True) as old_desc_inst,
        patch(_GET_SP_PLAN, return_value=shared_plan),
    ):
        old_count = _apply_sp_runtime(model_old, parallel_config)
        old_legacy_calls = list(old_legacy_inst.call_args_list)
        old_desc_calls = list(old_desc_inst.call_args_list)
        old_marker = fc_old.sp_plan_hooks_applied

    # ---- New path ----
    model_new = _build_pipeline()
    fc_new = _make_forward_context(parallel_config)
    with (
        override_forward_context(fc_new),
        patch(_NEW_PATH_INSTALLER) as new_legacy_inst,
        patch(_NEW_PATH_DESCRIPTOR_INSTALLER, return_value=True) as new_desc_inst,
        patch(_GET_SP_PLAN, return_value=shared_plan),
    ):
        new_count = apply_sp_to_pipeline(model_new, parallel_config)
        new_legacy_calls = list(new_legacy_inst.call_args_list)
        new_desc_calls = list(new_desc_inst.call_args_list)
        new_marker = fc_new.sp_plan_hooks_applied

    assert old_count == new_count == 2, (
        f"expected 2 applications on Wan22 pipeline, "
        f"got old={old_count} new={new_count}"
    )
    assert old_marker is True and new_marker is True
    assert len(old_legacy_calls) == len(new_legacy_calls) == 2
    assert len(old_desc_calls) == len(new_desc_calls) == 0

    # Adapter-ordering assertion: each path's call list MUST hit the
    # transformer slot first, then transformer_2, when keyed by
    # per-pipeline identity.
    def _ordering(calls: list, model: _FakeWan22Pipeline) -> list[bool]:
        return [c.args[0] is model.transformer for c in calls]

    assert _ordering(old_legacy_calls, model_old) == [True, False]
    assert _ordering(new_legacy_calls, model_new) == [True, False]

    # Full call_args_list equality (excluding positional 0 since the two
    # runs see different transformer instances).
    assert [_normalize(c) for c in old_legacy_calls] == [
        _normalize(c) for c in new_legacy_calls
    ]
    assert [_normalize(c) for c in old_desc_calls] == [
        _normalize(c) for c in new_desc_calls
    ]


# -----------------------------------------------------------------------------
# Test 3: 4-cell matrix (use_sp_descriptor × with_descriptor) on a single
#         DefaultAdapter-shaped pipeline. The (False, True) cell is the
#         F2 lock-in case.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "use_sp_descriptor,with_descriptor",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
    ids=[
        "plain_legacy",
        "F2_lockin_legacy_with_descriptor_present",
        "descriptor_flag_no_descriptor",
        "descriptor_flag_with_descriptor",
    ],
)
def test_twin_path_equivalence_4_cell_matrix(
    use_sp_descriptor: bool, with_descriptor: bool,
):
    """Exercises all 4 cells of the descriptor matrix.

    The ``(use_sp_descriptor=False, with_descriptor=True)`` cell is the
    F2 lock-in (``DESIGN_PHASE1C_INIT_DISPATCH.md`` §4.5.4): even when the
    model carries a descriptor, the legacy ``_sp_plan`` branch MUST be
    taken when the flag is OFF — both paths agree on this. The
    ``(use_sp_descriptor=True, with_descriptor=*)`` cells route through
    ``apply_sequence_parallel_from_descriptor``.
    """
    parallel_config = _make_parallel_config(
        ulysses_degree=2,
        ring_degree=1,
        use_sp_descriptor=use_sp_descriptor,
    )

    # Both paths must resolve the SAME plan object so the installer receives
    # an identical plan argument; MagicMock compares by identity, so two
    # separate instances would break the call_args_list equality below.
    shared_plan = MagicMock(name="fake_plan")

    # ---- Legacy path ----
    model_old = _FakePipeline(_FakeTransformer(with_descriptor=with_descriptor))
    fc_old = _make_forward_context(parallel_config)
    with (
        override_forward_context(fc_old),
        patch(_LEGACY_PATH_INSTALLER) as old_legacy_inst,
        patch(_LEGACY_PATH_DESCRIPTOR_INSTALLER, return_value=True) as old_desc_inst,
        patch(_GET_SP_PLAN, return_value=shared_plan),
    ):
        old_count = _apply_sp_runtime(model_old, parallel_config)
        old_legacy_calls = list(old_legacy_inst.call_args_list)
        old_desc_calls = list(old_desc_inst.call_args_list)
        old_marker = fc_old.sp_plan_hooks_applied

    # ---- New path ----
    model_new = _FakePipeline(_FakeTransformer(with_descriptor=with_descriptor))
    fc_new = _make_forward_context(parallel_config)
    with (
        override_forward_context(fc_new),
        patch(_NEW_PATH_INSTALLER) as new_legacy_inst,
        patch(_NEW_PATH_DESCRIPTOR_INSTALLER, return_value=True) as new_desc_inst,
        patch(_GET_SP_PLAN, return_value=shared_plan),
    ):
        new_count = apply_sp_to_pipeline(model_new, parallel_config)
        new_legacy_calls = list(new_legacy_inst.call_args_list)
        new_desc_calls = list(new_desc_inst.call_args_list)
        new_marker = fc_new.sp_plan_hooks_applied

    assert old_count == new_count, (
        f"[{use_sp_descriptor=}, {with_descriptor=}] applied_count "
        f"mismatch: old={old_count} new={new_count}"
    )
    assert old_marker == new_marker
    assert len(old_legacy_calls) == len(new_legacy_calls)
    assert len(old_desc_calls) == len(new_desc_calls)
    assert [_normalize(c) for c in old_legacy_calls] == [
        _normalize(c) for c in new_legacy_calls
    ]
    assert [_normalize(c) for c in old_desc_calls] == [
        _normalize(c) for c in new_desc_calls
    ]

    # F2 lock-in: when use_sp_descriptor=False, the descriptor installer
    # MUST NOT be called even if the model carries an ``_sp_descriptor``.
    if use_sp_descriptor is False:
        assert len(new_desc_calls) == 0, (
            "F2 lock-in violated: descriptor installer was called on "
            "the new path with use_sp_descriptor=False."
        )
        assert len(old_desc_calls) == 0


# -----------------------------------------------------------------------------
# Test 4: TWIN-3 Option A — adapter-specific no-target warning.
# -----------------------------------------------------------------------------


def test_no_target_warning_is_adapter_specific():
    """The new path's no-target warning is intentionally adapter-specific
    (TWIN-3 Option A): it points users at registering a
    :class:`PipelineTransformerAdapter` rather than reproducing the legacy
    "no hook-based SP plan/descriptor was applied" text. The equivalence
    gates do NOT assert log-text equality, so this test verifies the new
    path's actionable wording is present without comparing byte-by-byte
    against legacy.
    """
    import vllm_omni.diffusion.distributed.sp_runtime as sp_runtime_module

    parallel_config = _make_parallel_config(
        ulysses_degree=2, ring_degree=1, use_sp_descriptor=False,
    )

    pipeline = _NoTransformerPipeline()
    fc = _make_forward_context(parallel_config)
    with (
        override_forward_context(fc),
        patch.object(sp_runtime_module, "logger") as mock_logger,
    ):
        applied = apply_sp_to_pipeline(pipeline, parallel_config)

    assert applied == 0
    assert fc.sp_plan_hooks_applied is False, (
        "no-target case must NOT set sp_plan_hooks_applied (matches "
        "legacy semantics: marker tracks 'hooks actually registered')."
    )

    warning_calls = mock_logger.warning.call_args_list
    assert len(warning_calls) == 1, (
        f"expected exactly one warning, got {len(warning_calls)}"
    )
    msg = warning_calls[0].args[0]
    assert "PipelineTransformerAdapter" in msg, (
        "new-path no-target warning MUST mention PipelineTransformerAdapter "
        f"(TWIN-3 Option A); got: {msg!r}"
    )
    assert "_NoTransformerPipeline" in msg, (
        "new-path no-target warning SHOULD include the pipeline class "
        f"name for actionability; got: {msg!r}"
    )
