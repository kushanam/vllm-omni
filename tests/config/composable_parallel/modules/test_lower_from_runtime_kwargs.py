# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T2 (Phase 1c, §6.1.3): ``Orchestrator.lower_from_runtime_kwargs`` tests.

CPU-only. The reconstruction is a pure function of
``(od_config.parallel_config, execution_type)``. The contract this file
locks (per §4.1.2 / §4.1.4 / §5.2 O1):

* Returns an :class:`InitDispatchPlan` containing ONLY axes that are
  init-dispatchable for the active backend (capability ∩ ``delegated``) —
  never ``stage_replica`` / ``tp`` / ``dp`` / ``pp`` / ``ep``, even when the
  input ``parallel_config`` carries positive degrees for them. This is the
  F1 / Round-2 correctness anchor.
* Axes are active iff their degree field on ``DiffusionParallelConfig`` is
  strictly greater than 1 (``vae_patch_parallel_size``, ``ulysses_degree``,
  ``ring_degree``). Inert axes (degree==1) are absent from the plan.
* Reconstruction failures (missing attribute, non-numeric degree, etc.) are
  wrapped in :class:`RuntimePlanReconstructionError` (§4.1.5 case 3); no
  silent fallback.

We use :class:`SimpleNamespace` to stand in for ``OmniDiffusionConfig`` — the
reconstruction reads only three fields off ``parallel_config``, so a
fully-built ``OmniDiffusionConfig`` is unnecessary (and would drag in heavy
imports). This mirrors ``test_vae_pp_module.py`` which uses the same idiom.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.config.composable_parallel.backends import (
    VLLM_BACKEND,
    effective_init_dispatch_axes,
)
from vllm_omni.config.composable_parallel.modules.axes import (
    RingSequenceParallelStrategy,
    UlyssesSequenceParallelStrategy,
    VaePatchParallelStrategy,
)
from vllm_omni.config.composable_parallel.modules.base import LoweringCtx
from vllm_omni.config.composable_parallel.modules.orchestrator import (
    Orchestrator,
    RuntimePlanReconstructionError,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# The effective init-dispatch set for the vLLM backend (replaces the deleted
# ``INIT_DISPATCHABLE`` constant): capability ∩ backend.delegated.
INIT_DISPATCHABLE = effective_init_dispatch_axes(VLLM_BACKEND)

# Hard local regression signal independent of production metadata: pin the
# literal expected set so a drift in the modules/backend table is caught here
# even if the derived alias above silently changes.
assert INIT_DISPATCHABLE == frozenset({"vae_pp", "sp_ulysses", "sp_ring"}), (
    f"unexpected init-dispatch set for vLLM backend: {sorted(INIT_DISPATCHABLE)}"
)


# ---------------------------------------------------------------------------
# Helpers — minimal SimpleNamespace stand-in for OmniDiffusionConfig
# ---------------------------------------------------------------------------
def _make_od_config(
    *,
    vae_patch_parallel_size: int = 1,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    # Optional "noise" axes — the reconstruction MUST ignore them per §5.2 O1.
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    stage_replica_size: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            vae_patch_parallel_size=vae_patch_parallel_size,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
            stage_replica_size=stage_replica_size,
        ),
    )


def _axes_in(plan) -> list[str]:
    """Per-module ``axis`` list — the primary observable for these tests."""
    return [m.axis for m in plan]


def _plan_signatures(plan) -> list:
    """Frozen :class:`AxisPlan` signatures, used for deep-equal checks.

    The three INIT_DISPATCHABLE axes' ``plan()`` outputs are
    execution-type-invariant, so a bare :class:`LoweringCtx` is sufficient.
    """
    return [m.plan(LoweringCtx()) for m in plan]


# ---------------------------------------------------------------------------
# Happy-path single-axis cases (§6.1.3 axis coverage)
# ---------------------------------------------------------------------------
def test_returns_empty_plan_when_no_degrees_greater_than_one():
    od_config = _make_od_config()
    plan = Orchestrator().lower_from_runtime_kwargs(od_config, execution_type=None)
    assert plan == []


def test_returns_vae_pp_module_when_vae_patch_parallel_size_gt_1():
    od_config = _make_od_config(vae_patch_parallel_size=2)
    plan = Orchestrator().lower_from_runtime_kwargs(od_config, execution_type=None)
    assert _axes_in(plan) == ["vae_pp"]
    assert isinstance(plan[0], VaePatchParallelStrategy)
    # Sanity: plan()'s degree mirrors the input.
    assert _plan_signatures(plan)[0].degree == 2


def test_returns_sp_ulysses_module_when_ulysses_degree_gt_1():
    od_config = _make_od_config(ulysses_degree=2)
    plan = Orchestrator().lower_from_runtime_kwargs(od_config, execution_type=None)
    assert _axes_in(plan) == ["sp_ulysses"]
    assert isinstance(plan[0], UlyssesSequenceParallelStrategy)
    assert _plan_signatures(plan)[0].degree == 2


def test_returns_sp_ring_module_when_ring_degree_gt_1():
    od_config = _make_od_config(ring_degree=2)
    plan = Orchestrator().lower_from_runtime_kwargs(od_config, execution_type=None)
    assert _axes_in(plan) == ["sp_ring"]
    assert isinstance(plan[0], RingSequenceParallelStrategy)
    assert _plan_signatures(plan)[0].degree == 2


# ---------------------------------------------------------------------------
# Multi-axis cases (§6.1.3 mixed coverage)
# ---------------------------------------------------------------------------
def test_returns_sp_ulysses_and_sp_ring_when_both_degrees_gt_1():
    """Hybrid SP: both axes active. Reconstruction emits both modules in the
    candidate-build order (vae_pp, sp_ulysses, sp_ring); the canonical exec
    order is applied later by :meth:`Orchestrator.apply`."""
    od_config = _make_od_config(ulysses_degree=2, ring_degree=4)
    plan = Orchestrator().lower_from_runtime_kwargs(od_config, execution_type=None)
    assert set(_axes_in(plan)) == {"sp_ulysses", "sp_ring"}
    by_axis = {m.axis: m for m in plan}
    assert by_axis["sp_ulysses"].plan(LoweringCtx()).degree == 2
    assert by_axis["sp_ring"].plan(LoweringCtx()).degree == 4


def test_returns_vae_pp_and_sp_ulysses_when_both_active():
    """Mixed VAE-PP + SP: both axes active. The plan contains both modules
    (the canonical exec order is :data:`APPLY_ORDER`, applied by
    :meth:`Orchestrator.apply`, not by this reconstruction)."""
    od_config = _make_od_config(vae_patch_parallel_size=4, ulysses_degree=2)
    plan = Orchestrator().lower_from_runtime_kwargs(od_config, execution_type=None)
    assert set(_axes_in(plan)) == {"vae_pp", "sp_ulysses"}


def test_returns_all_three_when_every_dispatchable_axis_is_active():
    od_config = _make_od_config(
        vae_patch_parallel_size=2, ulysses_degree=2, ring_degree=2
    )
    plan = Orchestrator().lower_from_runtime_kwargs(od_config, execution_type=None)
    assert set(_axes_in(plan)) == INIT_DISPATCHABLE


# ---------------------------------------------------------------------------
# F1 / §5.2 O1 — the central correctness condition: NEVER returns non-dispatchable axes
# ---------------------------------------------------------------------------
def test_excludes_tp_dp_pp_ep_stage_replica_even_when_implied_by_config():
    """The runtime InitDispatchPlan MUST contain ONLY init-dispatchable axes,
    even when ``od_config.parallel_config`` carries positive degrees for
    ``tp``/``dp``/``pp``/``ep``/``stage_replica``. This is the F1 / Round-2
    correctness anchor — silently inventing modules the worker cannot honor
    is exactly what §4.1.2 / §5.2 O1 forbid."""
    od_config = _make_od_config(
        vae_patch_parallel_size=2,        # dispatchable, active
        ulysses_degree=2,                  # dispatchable, active
        ring_degree=1,                     # dispatchable, inert
        # The next five fields are intentionally > 1 / True. Reconstruction
        # MUST ignore them; they are non-dispatchable axes whose state lives
        # only on the config-time StrategyPlan.
        tensor_parallel_size=4,
        data_parallel_size=2,
        pipeline_parallel_size=2,
        enable_expert_parallel=True,
        stage_replica_size=3,
    )
    plan = Orchestrator().lower_from_runtime_kwargs(od_config, execution_type=None)
    # Only the two active dispatchable axes survive — and NOTHING ELSE.
    assert set(_axes_in(plan)) == {"vae_pp", "sp_ulysses"}
    for module in plan:
        assert module.axis in INIT_DISPATCHABLE


def test_returned_modules_are_owned_by_omni():
    """Every dispatchable axis is omni-executed (DelegatedStrategy axes
    cannot reach the runtime side; this guards against a future regression
    that adds a vLLM-owned axis to :data:`INIT_DISPATCHABLE` without an
    ``apply()`` impl)."""
    od_config = _make_od_config(
        vae_patch_parallel_size=2, ulysses_degree=2, ring_degree=2
    )
    plan = Orchestrator().lower_from_runtime_kwargs(od_config, execution_type=None)
    for module in plan:
        assert module.plan(LoweringCtx()).owned_by == "omni"


# ---------------------------------------------------------------------------
# §4.1.5 case 3 — fail-loud on reconstruction failure
# ---------------------------------------------------------------------------
def test_missing_parallel_config_attribute_raises_reconstruction_error():
    """A malformed ``od_config`` (no ``parallel_config``) cannot be parsed:
    reconstruction wraps the underlying error in
    :class:`RuntimePlanReconstructionError` and carries the offender."""
    od_config = SimpleNamespace()  # no parallel_config

    with pytest.raises(RuntimePlanReconstructionError) as exc_info:
        Orchestrator().lower_from_runtime_kwargs(od_config, execution_type=None)
    assert exc_info.value.od_config is od_config
    assert "parallel_config" in str(exc_info.value)


def test_missing_degree_field_raises_reconstruction_error():
    """``parallel_config`` exists but the required degree fields are absent
    — typical "corrupt config" shape. Fails LOUD (§4.1.5 case 3)."""
    od_config = SimpleNamespace(parallel_config=SimpleNamespace())  # missing degrees

    with pytest.raises(RuntimePlanReconstructionError) as exc_info:
        Orchestrator().lower_from_runtime_kwargs(od_config, execution_type=None)
    assert exc_info.value.od_config is od_config


def test_non_numeric_degree_raises_reconstruction_error():
    """A degree field that cannot be coerced to ``int`` is the §4.1.5
    "corrupt vae_patch_parallel_size" example. Fails LOUD."""
    od_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            vae_patch_parallel_size="not-a-number",
            ulysses_degree=1,
            ring_degree=1,
        ),
    )
    with pytest.raises(RuntimePlanReconstructionError) as exc_info:
        Orchestrator().lower_from_runtime_kwargs(od_config, execution_type=None)
    assert exc_info.value.od_config is od_config


def test_reconstruction_error_inherits_from_orchestrator_error():
    """Callers that ``except OrchestratorError`` also catch reconstruction
    failures (lets future code centralize orchestrator-error handling)."""
    from vllm_omni.config.composable_parallel.modules.orchestrator import (
        OrchestratorError,
    )

    assert issubclass(RuntimePlanReconstructionError, OrchestratorError)


# ---------------------------------------------------------------------------
# Execution-type parameter is forward-compat only (§4.1.2 docstring)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "execution_type",
    [None, "diffusion", "llm_ar", object()],
    ids=["none", "diffusion-str", "llm_ar-str", "opaque"],
)
def test_execution_type_does_not_alter_the_subplan_for_dispatchable_axes(execution_type):
    """Today's three dispatchable axes (``vae_pp`` / ``sp_ulysses`` /
    ``sp_ring``) are execution-type-invariant. Varying ``execution_type``
    must not affect the reconstructed subplan. (Future axes whose ``plan()``
    depends on execution_type would change this contract — at which point
    this test must be expanded with the new axis's expected behavior.)"""
    od_config = _make_od_config(
        vae_patch_parallel_size=2, ulysses_degree=2, ring_degree=2
    )
    plan = Orchestrator().lower_from_runtime_kwargs(
        od_config, execution_type=execution_type
    )
    assert set(_axes_in(plan)) == INIT_DISPATCHABLE
