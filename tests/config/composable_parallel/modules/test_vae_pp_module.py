# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T3 (CPU part): VAE-PP module plan() + apply() side-effect identity (§5.1, §5.3).

These assert that ``VaePatchParallelStrategy`` produces the SAME observable side
effects as the inline ``registry.initialize_model`` VAE-PP block did. Note that
the original ``set_parallel_size`` call was gated ONLY on ``is_distributed_vae``
(NOT on ``vae_pp_size > 1``), while the warning and the ``vae_use_tiling``
auto-enable are gated on ``n > 1``:
  - n=1, distributed VAE    -> set_parallel_size(1); NO warning, NO auto-enable
  - n=1, non-distributed    -> truly nothing
  - n>1, distributed VAE    -> auto-enable vae_use_tiling + set_parallel_size(n)
  - n>1, non-distributed    -> warning, NO set_parallel_size call
plus the §4.2 tiling-ordering nuance (auto-enable runs before the final
use_tiling assignment, so the auto-enabled value is what gets written).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import vllm_omni.config.composable_parallel.modules.axes.vae_pp as vae_pp_mod
from vllm_omni.config.composable_parallel.modules.axes.vae_pp import (
    VaePatchParallelStrategy,
)
from vllm_omni.config.composable_parallel.modules.base import (
    ApplyCtx,
    AxisPlan,
    LoweringCtx,
)
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedVaeMixin,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Verbatim message templates from the original registry vae_pp block. The
# captured-log helper applies ``%`` formatting, so these are the formatted texts.
_WARNING_TMPL = (
    "vae_patch_parallel_size=%d is set but VAE patch parallelism is NOT enabled "
    "for %s; ignoring."
)
_INFO_TMPL = (
    "vae_patch_parallel_size=%d requires vae_use_tiling; automatically enabling it."
)


class _RecordingDistributedVae(DistributedVaeMixin):
    """A DistributedVaeMixin VAE that records set_parallel_size calls.

    isinstance(vae, DistributedVaeMixin) is True (the registry's distributed
    check), but set_parallel_size is overridden to record rather than touch a
    real distributed executor.
    """

    def __init__(self) -> None:
        self.set_parallel_size_calls: list[int] = []
        # Memory-opt attributes (set by the registry's inline lines after apply()).
        self.use_slicing = False
        self.use_tiling = False

    def set_parallel_size(self, parallel_size: int) -> None:
        self.set_parallel_size_calls.append(parallel_size)


class _PlainVae:
    """A VAE that is NOT a DistributedVaeMixin (non-distributed path)."""

    def __init__(self) -> None:
        self.use_slicing = False
        self.use_tiling = False
        self.set_parallel_size_calls: list[int] = []

    def set_parallel_size(self, parallel_size: int) -> None:  # pragma: no cover
        self.set_parallel_size_calls.append(parallel_size)


def _make_od_config(vae_pp_size: int, vae_use_tiling: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(vae_patch_parallel_size=vae_pp_size),
        vae_use_tiling=vae_use_tiling,
        vae_use_slicing=False,
        model_class_name="FakePipeline",
    )


@pytest.fixture
def captured_logs(monkeypatch):
    """Capture the module logger's warning/info calls without a real handler."""
    warnings: list[str] = []
    infos: list[str] = []

    def _warning(msg, *args, **kwargs):
        warnings.append(msg % args if args else msg)

    def _info(msg, *args, **kwargs):
        infos.append(msg % args if args else msg)

    monkeypatch.setattr(vae_pp_mod.logger, "warning", _warning)
    monkeypatch.setattr(vae_pp_mod.logger, "info", _info)
    return SimpleNamespace(warnings=warnings, infos=infos)


# ---------------------------------------------------------------------------
# §5.1 — plan() assertion
# ---------------------------------------------------------------------------
def test_plan_matches_spec():
    plan = VaePatchParallelStrategy(2).plan(LoweringCtx())
    assert plan == AxisPlan(
        axis="vae_pp",
        degree=2,
        owned_by="omni",
        engine_kwargs={"vae_patch_parallel_size": 2},
        rank_token=None,
        consumes_world_dim=False,
    )


def test_build_groups_is_reused_noop():
    result = VaePatchParallelStrategy(2).build_groups(
        SimpleNamespace(rank_generator=object(), backend="nccl", world_size=4)
    )
    assert result.axis == "vae_pp"
    assert result.group is not None
    assert result.group.parallel_mode == "reused"
    assert result.group.reused is True
    assert result.group.coordinator is None


# ---------------------------------------------------------------------------
# §5.3 — apply() side-effect identity
# ---------------------------------------------------------------------------
def test_apply_n1_distributed_calls_set_parallel_size_only(captured_logs):
    vae = _RecordingDistributedVae()
    model = SimpleNamespace(vae=vae)
    od_config = _make_od_config(vae_pp_size=1, vae_use_tiling=False)

    VaePatchParallelStrategy(1).apply(ApplyCtx(model=model, od_config=od_config))

    # Original behavior: set_parallel_size is gated ONLY on is_distributed_vae,
    # so at n=1 a distributed VAE STILL gets set_parallel_size(1). The warning
    # and tiling auto-enable are gated on n>1, so neither fires.
    assert captured_logs.warnings == []
    assert captured_logs.infos == []
    assert od_config.vae_use_tiling is False
    assert vae.set_parallel_size_calls == [1]


def test_apply_n1_non_distributed_is_noop(captured_logs):
    vae = _PlainVae()
    model = SimpleNamespace(vae=vae)
    od_config = _make_od_config(vae_pp_size=1, vae_use_tiling=False)

    VaePatchParallelStrategy(1).apply(ApplyCtx(model=model, od_config=od_config))

    # n=1, non-distributed: truly nothing (warning gated n>1; not distributed).
    assert captured_logs.warnings == []
    assert captured_logs.infos == []
    assert od_config.vae_use_tiling is False
    assert vae.set_parallel_size_calls == []


def test_apply_n_gt1_distributed_auto_enables_tiling_and_sets_parallel_size(captured_logs):
    vae = _RecordingDistributedVae()
    model = SimpleNamespace(vae=vae)
    od_config = _make_od_config(vae_pp_size=4, vae_use_tiling=False)

    VaePatchParallelStrategy(4).apply(ApplyCtx(model=model, od_config=od_config))

    assert captured_logs.warnings == []
    assert captured_logs.infos == [_INFO_TMPL % 4]
    assert od_config.vae_use_tiling is True
    assert vae.set_parallel_size_calls == [4]


def test_apply_n_gt1_distributed_tiling_already_on_no_info(captured_logs):
    vae = _RecordingDistributedVae()
    model = SimpleNamespace(vae=vae)
    od_config = _make_od_config(vae_pp_size=4, vae_use_tiling=True)

    VaePatchParallelStrategy(4).apply(ApplyCtx(model=model, od_config=od_config))

    # tiling already on: no auto-enable info, but still set_parallel_size(n).
    assert captured_logs.warnings == []
    assert captured_logs.infos == []
    assert od_config.vae_use_tiling is True
    assert vae.set_parallel_size_calls == [4]


def test_apply_n_gt1_non_distributed_warns_no_set_parallel_size(captured_logs):
    vae = _PlainVae()
    model = SimpleNamespace(vae=vae)
    od_config = _make_od_config(vae_pp_size=4, vae_use_tiling=False)

    VaePatchParallelStrategy(4).apply(ApplyCtx(model=model, od_config=od_config))

    # Non-distributed VAE: warn, do NOT auto-enable tiling, do NOT set_parallel_size.
    assert captured_logs.warnings == [_WARNING_TMPL % (4, "FakePipeline")]
    assert captured_logs.infos == []
    assert od_config.vae_use_tiling is False
    assert vae.set_parallel_size_calls == []


# ---------------------------------------------------------------------------
# §4.2 — tiling-ordering nuance
# ---------------------------------------------------------------------------
def test_tiling_auto_enable_before_final_use_tiling_assignment(captured_logs):
    """apply() auto-enables vae_use_tiling BEFORE the registry writes
    model.vae.use_tiling = od_config.vae_use_tiling, so the auto-enabled value
    is what propagates to the VAE (§4.2)."""
    vae = _RecordingDistributedVae()
    model = SimpleNamespace(vae=vae)
    od_config = _make_od_config(vae_pp_size=4, vae_use_tiling=False)

    # 1. Module apply() (auto-enable + set_parallel_size) — runs FIRST in registry.
    VaePatchParallelStrategy(4).apply(ApplyCtx(model=model, od_config=od_config))
    assert od_config.vae_use_tiling is True  # auto-enabled by apply()

    # 2. Registry memory-opt assignment runs AFTER apply(), reading the now-True value.
    if hasattr(model.vae, "use_tiling"):
        model.vae.use_tiling = od_config.vae_use_tiling

    assert model.vae.use_tiling is True
