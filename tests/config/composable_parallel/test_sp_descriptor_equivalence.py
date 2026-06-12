# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Descriptor<->legacy equivalence tests for the Phase-1b SP pilots.

The lock that makes flag-ON byte-identical for QwenImage: its ``_sp_descriptor``
must expand to a plan that deep-equals the legacy ``_sp_plan`` dict. Also asserts
BAGEL's marker is ``SPInternal`` and that the thin descriptor entrypoint behaves
as specified (no hooks for SPInternal; the same plan for an SPDescriptor model).

These import real model classes (diffusers / vLLM), so they ``importorskip`` and
are skipped gracefully where those deps are absent.
"""
from __future__ import annotations

from unittest import mock

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# The declaration layer is torch-free; the model classes are not.
torch = pytest.importorskip("torch")

from vllm_omni.diffusion.distributed.sp_descriptor import SPDescriptor, SPInternal  # noqa: E402
from vllm_omni.diffusion.distributed.sp_plan import (  # noqa: E402
    SequenceParallelInput,
    SequenceParallelOutput,
    validate_sp_plan,
)


def _qwen_image_cls():
    mod = pytest.importorskip(
        "vllm_omni.diffusion.models.qwen_image.qwen_image_transformer"
    )
    return mod.QwenImageTransformer2DModel


def _bagel_cls():
    mod = pytest.importorskip("vllm_omni.diffusion.models.bagel.bagel_transformer")
    return mod.Bagel


def test_qwen_image_descriptor_to_plan_deep_equals_legacy_sp_plan():
    cls = _qwen_image_cls()
    desc: SPDescriptor = cls._sp_descriptor
    assert isinstance(desc, SPDescriptor)

    # QwenImage's descriptor is fully static (no builder/parent), so model is
    # unused by to_plan(); pass None.
    produced = desc.to_plan(model=None)
    legacy = cls._sp_plan

    # Same module keys.
    assert set(produced.keys()) == set(legacy.keys())

    # Deep-equal on each module's spec (dict of SequenceParallelInput, or a bare
    # SequenceParallelOutput). dataclasses are frozen + comparable by value.
    assert produced == legacy


def test_qwen_image_descriptor_edge_cases_match_legacy_plan():
    """Edge-case lock for the QwenImage pilot (not just a happy-path deep-equal).

    QwenImage exercises the non-trivial descriptor features: a *sparse* output
    split (``modulate_index_prepare`` shards only output index 1, leaving index 0
    replicated), per-index ``split_dim`` (idx0 dim=1 vs idx1 dim=0), ``auto_pad``,
    and a single-output gather that must collapse to a bare
    ``SequenceParallelOutput``. The compiled plan must reproduce all of these and
    must pass the existing runtime validator.
    """
    cls = _qwen_image_cls()
    produced = cls._sp_descriptor.to_plan(model=None)

    # The compiled plan must satisfy the existing runtime validator.
    validate_sp_plan(produced)

    # Multi-output split with per-index split_dim + auto_pad preserved.
    rope = produced["image_rope_prepare"]
    assert isinstance(rope, dict)
    assert rope[0] == SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True)
    assert rope[1] == SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True, auto_pad=True)
    # Output index 2 (txt_freqs) is intentionally NOT sharded -> absent from the dict.
    assert set(rope.keys()) == {0, 1}

    # Sparse conditional split: only output index 1 of modulate_index_prepare is
    # sharded (index 0 stays replicated), i.e. a split dict keyed solely by 1.
    modulate = produced["modulate_index_prepare"]
    assert isinstance(modulate, dict)
    assert set(modulate.keys()) == {1}
    assert modulate[1] == SequenceParallelInput(split_dim=1, expected_dims=2, split_output=True, auto_pad=True)

    # Single-output gather collapses to a bare SequenceParallelOutput (not a list).
    assert produced["proj_out"] == SequenceParallelOutput(gather_dim=1, expected_dims=3)


def test_bagel_marker_is_sp_internal():
    cls = _bagel_cls()
    assert isinstance(cls._sp_descriptor, SPInternal)


def test_apply_from_descriptor_noops_on_sp_internal():
    from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
    from vllm_omni.diffusion.hooks import sequence_parallel as sp_hooks

    model = mock.MagicMock()
    model._sp_descriptor = SPInternal("manual SP in forward")
    config = SequenceParallelConfig(ulysses_degree=2, ring_degree=1)

    with mock.patch.object(sp_hooks, "apply_sequence_parallel") as apply_spy:
        applied = sp_hooks.apply_sequence_parallel_from_descriptor(model, config)

    assert applied is False
    apply_spy.assert_not_called()


def test_apply_from_descriptor_uses_legacy_plan_for_sp_descriptor_model():
    from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
    from vllm_omni.diffusion.hooks import sequence_parallel as sp_hooks

    cls = _qwen_image_cls()
    expected_plan = cls._sp_descriptor.to_plan(model=None)

    # A stand-in model carrying the real descriptor as an attribute.
    model = mock.MagicMock()
    model._sp_descriptor = cls._sp_descriptor
    config = SequenceParallelConfig(ulysses_degree=2, ring_degree=1)

    with mock.patch.object(sp_hooks, "apply_sequence_parallel") as apply_spy:
        applied = sp_hooks.apply_sequence_parallel_from_descriptor(model, config)

    assert applied is True
    apply_spy.assert_called_once()
    # apply_sequence_parallel(model, config, plan) — the plan must equal the
    # legacy dict (== QwenImage._sp_plan).
    _, called_args, _ = apply_spy.mock_calls[0]
    assert called_args[2] == expected_plan
    assert called_args[2] == cls._sp_plan


def test_apply_from_descriptor_returns_false_when_no_descriptor_and_no_plan():
    from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
    from vllm_omni.diffusion.hooks import sequence_parallel as sp_hooks

    # An object with neither _sp_descriptor nor _sp_plan.
    class _Bare:
        pass

    config = SequenceParallelConfig(ulysses_degree=2, ring_degree=1)
    with mock.patch.object(sp_hooks, "apply_sequence_parallel") as apply_spy:
        applied = sp_hooks.apply_sequence_parallel_from_descriptor(_Bare(), config)

    assert applied is False
    apply_spy.assert_not_called()
