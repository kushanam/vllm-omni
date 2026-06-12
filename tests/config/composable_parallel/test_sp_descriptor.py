# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the typed SPDescriptor declaration layer (Phase 1b).

Covers descriptor->plan expansion, the escape hatches (parent/drop_modules,
builder), the SPInternal no-op marker, and the ``split_outputs`` helper. The
declaration layer is torch-free, so these run on plain CPU with no GPU/deps.
"""
from __future__ import annotations

import pytest

from vllm_omni.diffusion.distributed.sp_descriptor import (
    GatherSpec,
    SPDescriptor,
    SPInternal,
    SplitSpec,
    split_outputs,
)
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
    validate_sp_plan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_splitspec_input_target_to_runtime():
    # str target => input split (split_output False).
    spec = SplitSpec("", "hidden_states", split_dim=1, expected_dims=3)
    assert spec.is_output_split is False
    rt = spec.to_runtime()
    assert isinstance(rt, SequenceParallelInput)
    assert (rt.split_dim, rt.expected_dims, rt.split_output, rt.auto_pad) == (1, 3, False, False)


def test_splitspec_output_target_to_runtime():
    # int target => output split (split_output True).
    spec = SplitSpec("rope", 0, split_dim=0, expected_dims=2, auto_pad=True)
    assert spec.is_output_split is True
    rt = spec.to_runtime()
    assert (rt.split_dim, rt.expected_dims, rt.split_output, rt.auto_pad) == (0, 2, True, True)


def test_gatherspec_to_runtime():
    rt = GatherSpec("proj_out", gather_dim=1, expected_dims=3).to_runtime()
    assert isinstance(rt, SequenceParallelOutput)
    assert (rt.gather_dim, rt.expected_dims) == (1, 3)


def test_to_plan_single_output_gather_collapses_to_bare_output():
    desc = SPDescriptor(
        splits=(SplitSpec("rope", 0, split_dim=1, expected_dims=4, auto_pad=True),),
        gathers=(GatherSpec("proj_out", gather_dim=1, expected_dims=3),),
    )
    plan = desc.to_plan(model=None)
    # Split module -> dict keyed by target.
    assert plan["rope"] == {0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True, auto_pad=True)}
    # Single gather at index 0 collapses to a bare SequenceParallelOutput.
    assert plan["proj_out"] == SequenceParallelOutput(gather_dim=1, expected_dims=3)
    validate_sp_plan(plan)


def test_to_plan_multi_output_gather_is_position_indexed_list():
    # Gathers at indices 0 and 2 -> [out0, None, out2].
    desc = SPDescriptor(
        gathers=(
            GatherSpec("proj_out", gather_dim=1, expected_dims=3, output_index=0),
            GatherSpec("proj_out", gather_dim=1, expected_dims=3, output_index=2),
        ),
    )
    plan = desc.to_plan(model=None)
    out = plan["proj_out"]
    assert isinstance(out, list)
    assert out == [
        SequenceParallelOutput(gather_dim=1, expected_dims=3),
        None,
        SequenceParallelOutput(gather_dim=1, expected_dims=3),
    ]
    # SHOULD-FIX (review): the sparse multi-output gather plan the compiler emits
    # (with None holes) MUST pass the existing validator, since the runtime
    # gather hook already accepts None placeholders. This assertion is what
    # caught the to_plan()/validate_sp_plan() mismatch.
    validate_sp_plan(plan)


def test_to_plan_sparse_single_gather_at_nonzero_index_validates():
    # A single gather at a non-zero index does NOT collapse to a bare output;
    # it becomes [None, ..., out] with leading None holes and must validate.
    desc = SPDescriptor(
        gathers=(GatherSpec("proj_out", gather_dim=1, expected_dims=3, output_index=2),),
    )
    plan = desc.to_plan(model=None)
    out = plan["proj_out"]
    assert isinstance(out, list)
    assert out == [None, None, SequenceParallelOutput(gather_dim=1, expected_dims=3)]
    validate_sp_plan(plan)


def test_validate_rejects_all_none_output_list():
    # A degenerate output list with no real gather is malformed and must be
    # rejected (guards the relaxed None-placeholder rule from over-accepting).
    with pytest.raises(ValueError):
        validate_sp_plan({"proj_out": [None, None]})


def test_to_plan_mixed_split_and_gather_same_module_raises():
    # NIT (review): user-facing validation must raise ValueError (not assert).
    desc = SPDescriptor(
        splits=(SplitSpec("blocks.0", 0, split_dim=1, expected_dims=3),),
        gathers=(GatherSpec("blocks.0", gather_dim=1, expected_dims=3),),
    )
    with pytest.raises(ValueError, match="mixes split and gather"):
        desc.to_plan(model=None)


def test_to_plan_none_expected_dims_and_auto_pad_roundtrip():
    # Edge cases: expected_dims=None (no runtime dim check) and auto_pad on a
    # plain input split. Both must survive expansion and validate.
    desc = SPDescriptor(
        splits=(SplitSpec("", "hidden_states", split_dim=1, expected_dims=None, auto_pad=True),),
        gathers=(GatherSpec("proj_out", gather_dim=1, expected_dims=None),),
    )
    plan = desc.to_plan(model=None)
    inp = plan[""]["hidden_states"]
    assert isinstance(inp, SequenceParallelInput)
    assert inp.expected_dims is None
    assert inp.auto_pad is True
    assert inp.split_output is False
    assert plan["proj_out"] == SequenceParallelOutput(gather_dim=1, expected_dims=None)
    validate_sp_plan(plan)


def test_to_plan_per_index_split_dim_preserved():
    # Two outputs of the same module with DIFFERENT split_dim.
    desc = SPDescriptor(
        splits=(
            SplitSpec("prep", 0, split_dim=1, expected_dims=3, auto_pad=True),
            SplitSpec("prep", 1, split_dim=0, expected_dims=2, auto_pad=True),
        ),
    )
    plan = desc.to_plan(model=None)
    assert plan["prep"][0].split_dim == 1
    assert plan["prep"][1].split_dim == 0
    validate_sp_plan(plan)


def test_parent_drop_modules_inheritance():
    parent = SPDescriptor(
        splits=(
            SplitSpec("blocks.0", "hidden_states", split_dim=1, expected_dims=3),
            SplitSpec("rope", 0, split_dim=1, expected_dims=4),
        ),
        gathers=(GatherSpec("proj_out", gather_dim=1, expected_dims=3),),
    )
    child = SPDescriptor(
        parent=parent,
        drop_modules=("blocks.0",),
        splits=(SplitSpec("_sp_shard_point", 0, split_dim=1, expected_dims=3, auto_pad=True),),
    )
    resolved = child.resolve(model=None)
    modules = {s.module for s in resolved.splits}
    assert "blocks.0" not in modules  # dropped
    assert "rope" in modules  # inherited
    assert "_sp_shard_point" in modules  # added by child
    # parent gather inherited.
    assert any(g.module == "proj_out" for g in resolved.gathers)
    # parent/drop_modules collapsed away after resolve.
    assert resolved.parent is None and resolved.drop_modules == ()


def test_builder_resolution():
    built = SPDescriptor(splits=(SplitSpec("", "hidden_states", split_dim=1, expected_dims=3),))
    desc = SPDescriptor(builder=lambda model: built)
    resolved = desc.resolve(model=None)
    assert resolved.splits == built.splits
    # to_plan goes through the builder too.
    assert "" in desc.to_plan(model=None)


def test_split_outputs_helper():
    specs = split_outputs("rope", (0, 1), dim=1, dims=4, auto_pad=True)
    assert [s.target for s in specs] == [0, 1]
    assert all(s.module == "rope" and s.split_dim == 1 and s.expected_dims == 4 and s.auto_pad for s in specs)
    assert all(s.is_output_split for s in specs)


def test_sp_internal_marker():
    marker = SPInternal("note here")
    assert isinstance(marker, SPInternal)
    assert marker.note == "note here"
    # SPInternal is NOT an SPDescriptor.
    assert not isinstance(marker, SPDescriptor)
