# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T4a (Phase 1c §6.1.4): registry-level dispatch gate test — SP half.

CPU-only. Patches ``Orchestrator.apply`` to a sentinel and asserts the
``use_init_dispatch`` flag controls whether ``registry.initialize_model``
reaches the dispatch loop:

* ``use_init_dispatch=False`` → ``Orchestrator.apply`` MUST NOT be called;
  the legacy ``_apply_sequence_parallel_if_enabled`` wrapper IS called.
* ``use_init_dispatch=True``  → ``Orchestrator.apply`` MUST be called
  exactly once with an :class:`ApplyCtx` carrying the constructed
  ``model``, the ``od_config``, and the resolved ``execution_type``
  (``StageExecutionType.DIFFUSION`` per §10 Assumption 1); the legacy
  ``_apply_sequence_parallel_if_enabled`` wrapper MUST NOT be called.

This file is **co-owned by T3 (SP half) and T4 (VAE half)** per §6.1.4 / §8.
T3 contributes the SP-relevant assertions (the orchestrator call gate,
the SP-helper-not-called gate). T4 will append VAE-PP-specific assertions
(the inline VAE-PP block deletion gate, the dispatched ``vae_pp.apply``
call gate) when its dispatch-deletion patch lands. Mock everything below
``Orchestrator()``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from vllm_omni.config.composable_parallel.modules.base import ApplyCtx
from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakePipeline(nn.Module):
    """Tiny ``nn.Module`` that ``initialize_model`` will treat as the loaded
    pipeline. No ``vae`` attribute, no ``transformer`` attribute — the
    inline VAE-PP poke is benign on a vae-less model (vae_pp_size==1
    short-circuits before touching anything), and SP routing is mocked at
    the registry boundary anyway.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig | None = None) -> None:
        super().__init__()
        # Mirror the constructor signature ``model_class(od_config=od_config)``
        # at registry.py:345; the value is ignored for this test.
        self._od_config = od_config


def _fake_model_class(od_config: OmniDiffusionConfig) -> _FakePipeline:
    """Stand-in for the model class returned by
    ``DiffusionModelRegistry._try_load_model_cls``. Implemented as a
    callable, not a ``type``, so the tests can avoid quant-config plumbing
    (which calls ``current_omni_platform.get_diffusion_packed_modules_mapping``
    on a real class).
    """
    return _FakePipeline(od_config=od_config)


def _make_od_config(
    *,
    use_init_dispatch: bool,
    ulysses_degree: int = 2,
) -> OmniDiffusionConfig:
    """Diffusion config for the dispatch-gate tests.

    ``ulysses_degree=2`` activates SP so the orchestrator's runtime
    reconstruction (`lower_from_runtime_kwargs`) returns a non-empty plan
    when the flag is ON; that lets the test verify a real plan reaches
    ``Orchestrator.apply``. Quant config is intentionally None to avoid
    the real ``configure_quant_config`` path.
    """
    od = OmniDiffusionConfig()
    od.parallel_config = DiffusionParallelConfig(
        ulysses_degree=ulysses_degree,
        ring_degree=1,
        use_init_dispatch=use_init_dispatch,
    )
    od.model_class_name = "FakePipeline"
    od.quantization_config = None
    return od


# ---------------------------------------------------------------------------
# Tests — SP half (T3 contribution to T4a)
# ---------------------------------------------------------------------------
def test_off_path_does_not_invoke_orchestrator_apply():
    """``use_init_dispatch=False`` → ``Orchestrator.apply`` MUST NOT be
    called; the legacy SP wrapper IS called.

    This locks the §5.4 R1(a) flag-gating: the OFF path is byte-identical
    to the pre-1c inline path for the SP routing decision.
    """
    from vllm_omni.diffusion import registry  # local import: side-effect free

    od = _make_od_config(use_init_dispatch=False)

    with (
        patch.object(
            registry.DiffusionModelRegistry,
            "_try_load_model_cls",
            return_value=_fake_model_class,
        ),
        patch.object(
            registry,
            "_prepare_diffusion_quant_config",
            autospec=True,
        ),
        patch(
            "vllm_omni.config.composable_parallel.modules.orchestrator.Orchestrator.apply",
            autospec=True,
        ) as orch_apply,
        patch.object(
            registry,
            "_apply_sequence_parallel_if_enabled",
            autospec=True,
        ) as legacy_sp,
    ):
        registry.initialize_model(od)

    assert orch_apply.call_count == 0, (
        "OFF path: Orchestrator.apply MUST NOT be called when "
        "use_init_dispatch=False (Phase 1c §5.4 R1 flag gate)."
    )
    assert legacy_sp.call_count == 1, (
        "OFF path: legacy _apply_sequence_parallel_if_enabled MUST be "
        "called exactly once."
    )


def test_on_path_invokes_orchestrator_apply_with_correct_apply_ctx():
    """``use_init_dispatch=True`` → ``Orchestrator.apply`` IS called exactly
    once with an :class:`ApplyCtx` carrying the constructed ``model``, the
    ``od_config``, and ``execution_type=StageExecutionType.DIFFUSION``
    (§10 Assumption 1). The legacy SP wrapper is NOT called.
    """
    from vllm_omni.diffusion import registry  # local import: side-effect free

    od = _make_od_config(use_init_dispatch=True)
    captured: dict = {}

    # Capture the model returned by the model class so we can assert
    # ``ApplyCtx.model`` identity. The fake class returns a fresh
    # ``_FakePipeline`` per call; we record the latest one.
    def _capturing_model_class(od_config: OmniDiffusionConfig) -> _FakePipeline:
        m = _FakePipeline(od_config=od_config)
        captured["model"] = m
        return m

    sentinel_plan = MagicMock(name="sentinel_plan")
    sentinel_results: list = []

    with (
        patch.object(
            registry.DiffusionModelRegistry,
            "_try_load_model_cls",
            return_value=_capturing_model_class,
        ),
        patch.object(
            registry,
            "_prepare_diffusion_quant_config",
            autospec=True,
        ),
        patch(
            "vllm_omni.config.composable_parallel.modules.orchestrator."
            "Orchestrator.lower_from_runtime_kwargs",
            autospec=True,
            return_value=sentinel_plan,
        ) as lower_kwargs,
        patch(
            "vllm_omni.config.composable_parallel.modules.orchestrator.Orchestrator.apply",
            autospec=True,
            return_value=sentinel_results,
        ) as orch_apply,
        patch.object(
            registry,
            "_apply_sequence_parallel_if_enabled",
            autospec=True,
        ) as legacy_sp,
    ):
        registry.initialize_model(od)

    assert legacy_sp.call_count == 0, (
        "ON path: legacy _apply_sequence_parallel_if_enabled MUST NOT be "
        "called when use_init_dispatch=True; SP routing is reached via "
        "Orchestrator.apply."
    )
    assert orch_apply.call_count == 1, (
        "ON path: Orchestrator.apply MUST be called exactly once when "
        "use_init_dispatch=True (Phase 1c §5.4 R1 flag gate)."
    )

    # ``autospec=True`` makes the call args ``(self, plan, ctx)``. Pull the
    # ApplyCtx out and verify its fields match the spec's §4.3 contract.
    args, kwargs = orch_apply.call_args
    plan_arg = args[1] if len(args) >= 2 else kwargs.get("plan")
    ctx_arg = args[2] if len(args) >= 3 else kwargs.get("ctx")
    assert plan_arg is sentinel_plan, (
        "Orchestrator.apply must receive the plan returned by "
        "lower_from_runtime_kwargs."
    )
    assert isinstance(ctx_arg, ApplyCtx)
    assert ctx_arg.model is captured["model"], (
        "ApplyCtx.model must be the just-constructed pipeline."
    )
    assert ctx_arg.od_config is od, (
        "ApplyCtx.od_config must be the OmniDiffusionConfig passed to "
        "initialize_model."
    )
    assert ctx_arg.execution_type is StageExecutionType.DIFFUSION, (
        "ApplyCtx.execution_type must default to "
        "StageExecutionType.DIFFUSION (§10 Assumption 1; registry.initialize_model "
        "is only called for diffusion stages)."
    )

    # And lower_from_runtime_kwargs should have been called with the same
    # ``od_config`` + ``exec_type`` so the plan reaches the dispatcher.
    assert lower_kwargs.call_count == 1
    lk_args, lk_kwargs = lower_kwargs.call_args
    # Positional shape under autospec is (self, od_config, execution_type).
    assert lk_args[1] is od
    assert lk_args[2] is StageExecutionType.DIFFUSION


def test_on_path_uses_orchestrator_for_sp_active_config():
    """``use_init_dispatch=True`` with SP-active degrees → the orchestrator
    call path is the only SP-routing surface. Together with the F2 lock-in
    test in ``test_sequence_parallel.py``, this guarantees the (ON,
    use_sp_descriptor=False) cell of §4.5.4 routes through the dispatched
    ``sp_ulysses.apply`` / ``sp_ring.apply`` (which then call
    ``_apply_sp_runtime``) — never through the inline registry helper.

    The deeper SP-mechanism assertion (descriptor vs legacy ``_sp_plan``
    branch) is locked at the helper boundary by the F2 test in
    ``tests/diffusion/distributed/test_sequence_parallel.py``; here we
    only care that the SP call path is the orchestrator's.
    """
    from vllm_omni.diffusion import registry  # local import: side-effect free

    od = _make_od_config(use_init_dispatch=True, ulysses_degree=2)

    with (
        patch.object(
            registry.DiffusionModelRegistry,
            "_try_load_model_cls",
            return_value=_fake_model_class,
        ),
        patch.object(
            registry,
            "_prepare_diffusion_quant_config",
            autospec=True,
        ),
        patch(
            "vllm_omni.config.composable_parallel.modules.orchestrator."
            "Orchestrator.lower_from_runtime_kwargs",
            autospec=True,
            return_value=[],
        ),
        patch(
            "vllm_omni.config.composable_parallel.modules.orchestrator.Orchestrator.apply",
            autospec=True,
        ) as orch_apply,
        patch.object(
            registry,
            "_apply_sequence_parallel_if_enabled",
            autospec=True,
        ) as legacy_sp,
    ):
        registry.initialize_model(od)

    assert orch_apply.call_count == 1
    assert legacy_sp.call_count == 0


# ---------------------------------------------------------------------------
# Tests — VAE-PP half (T4 contribution to T4a)
# ---------------------------------------------------------------------------
# These VAE-PP-specific assertions complete the §6.1.4 dispatch-gate
# contract and lock the §5.4 R1 / §4.7 deletion of the bespoke VAE-PP call
# site at the pre-T4 ``registry.py:355-362``. T3's SP-half tests (above)
# pin the SP routing decision; T4's tests pin the VAE-PP routing decision.
# Together they assert the full dispatch-gate matrix: VAE-PP and SP each
# go through Orchestrator.apply on flag ON, and through their legacy
# inline call sites on flag OFF.
# ---------------------------------------------------------------------------
def _make_od_config_with_vae_pp(
    *,
    use_init_dispatch: bool,
    vae_patch_parallel_size: int = 2,
) -> OmniDiffusionConfig:
    """VAE-PP-active diffusion config for the dispatch-gate tests.

    ``vae_patch_parallel_size=2`` activates the VAE-PP axis so the
    orchestrator's runtime reconstruction (`lower_from_runtime_kwargs`)
    returns a non-empty plan when the flag is ON; that lets the test
    verify a real VAE-PP module reaches ``Orchestrator.apply``. Quant
    config is intentionally None to avoid the real
    ``configure_quant_config`` path.
    """
    od = OmniDiffusionConfig()
    od.parallel_config = DiffusionParallelConfig(
        ulysses_degree=1,
        ring_degree=1,
        vae_patch_parallel_size=vae_patch_parallel_size,
        use_init_dispatch=use_init_dispatch,
    )
    od.model_class_name = "FakePipeline"
    od.quantization_config = None
    return od


def test_off_path_vae_pp_invoked_via_bespoke_call():
    """``use_init_dispatch=False`` with VAE-PP active → the bespoke inline
    ``VaePatchParallelStrategy.apply`` call IS reached; ``Orchestrator.apply``
    is NOT called.

    Locks the §5.4 R1(a) "else run the existing inline paths verbatim"
    contract for VAE-PP: on the OFF path the today-verbatim sequence
    (vae_pp -> memory-opt -> legacy SP) is preserved, byte-identical to
    pre-1c (G5).
    """
    from vllm_omni.diffusion import registry  # local import: side-effect free

    od = _make_od_config_with_vae_pp(use_init_dispatch=False)

    with (
        patch.object(
            registry.DiffusionModelRegistry,
            "_try_load_model_cls",
            return_value=_fake_model_class,
        ),
        patch.object(
            registry,
            "_prepare_diffusion_quant_config",
            autospec=True,
        ),
        patch(
            "vllm_omni.config.composable_parallel.modules.orchestrator.Orchestrator.apply",
            autospec=True,
        ) as orch_apply,
        patch(
            "vllm_omni.config.composable_parallel.modules.axes.vae_pp."
            "VaePatchParallelStrategy.apply",
            autospec=True,
        ) as vae_pp_apply,
        patch.object(
            registry,
            "_apply_sequence_parallel_if_enabled",
            autospec=True,
        ),
    ):
        registry.initialize_model(od)

    assert orch_apply.call_count == 0, (
        "OFF path: Orchestrator.apply MUST NOT be called when "
        "use_init_dispatch=False (Phase 1c §5.4 R1 flag gate)."
    )
    assert vae_pp_apply.call_count == 1, (
        "OFF path: bespoke VaePatchParallelStrategy.apply MUST be called "
        "exactly once (the legacy inline VAE-PP call site)."
    )

    # ``autospec=True`` on a method makes the call args ``(self, ctx)``.
    # Verify the bespoke call passes an ApplyCtx carrying the constructed
    # model + the OmniDiffusionConfig (matching the legacy registry shape).
    args, kwargs = vae_pp_apply.call_args
    ctx_arg = args[1] if len(args) >= 2 else kwargs.get("ctx")
    assert isinstance(ctx_arg, ApplyCtx)
    assert ctx_arg.od_config is od, (
        "Bespoke VAE-PP call must receive the OmniDiffusionConfig "
        "passed to initialize_model."
    )


def test_on_path_vae_pp_invoked_via_dispatch_loop():
    """``use_init_dispatch=True`` with VAE-PP active → ``Orchestrator.apply``
    is called; the bespoke inline ``VaePatchParallelStrategy.apply`` is NOT
    reached. The orchestrator's runtime plan reconstruction
    (`lower_from_runtime_kwargs`) MUST include a ``VaePatchParallelStrategy``
    instance carrying the active ``vae_patch_parallel_size`` so the
    dispatched VAE-PP runs with the configured degree.

    Locks the §5.4 R1 / §4.7 deletion of the bespoke VAE-PP call site:
    after T4 lands, VAE-PP is reached on the ON path EXCLUSIVELY through
    ``Orchestrator.apply``; the inline ``VaePatchParallelStrategy(...).apply(...)``
    invocation no longer runs unconditionally on every init.
    """
    from vllm_omni.config.composable_parallel.modules.axes.vae_pp import (
        VaePatchParallelStrategy,
    )
    from vllm_omni.diffusion import registry  # local import: side-effect free

    od = _make_od_config_with_vae_pp(
        use_init_dispatch=True, vae_patch_parallel_size=2
    )

    with (
        patch.object(
            registry.DiffusionModelRegistry,
            "_try_load_model_cls",
            return_value=_fake_model_class,
        ),
        patch.object(
            registry,
            "_prepare_diffusion_quant_config",
            autospec=True,
        ),
        # NOTE: ``Orchestrator.apply`` is patched so the test runs CPU-only —
        # the real apply() would call ``VaePatchParallelStrategy.apply`` with
        # the dispatched plan. Asserting the plan reaches the orchestrator
        # with the right module is sufficient to lock the dispatch contract.
        patch(
            "vllm_omni.config.composable_parallel.modules.orchestrator.Orchestrator.apply",
            autospec=True,
        ) as orch_apply,
        patch(
            "vllm_omni.config.composable_parallel.modules.axes.vae_pp."
            "VaePatchParallelStrategy.apply",
            autospec=True,
        ) as vae_pp_apply,
        patch.object(
            registry,
            "_apply_sequence_parallel_if_enabled",
            autospec=True,
        ) as legacy_sp,
    ):
        registry.initialize_model(od)

    assert vae_pp_apply.call_count == 0, (
        "ON path: bespoke VaePatchParallelStrategy.apply MUST NOT be "
        "called when use_init_dispatch=True. VAE-PP is reached only via "
        "Orchestrator.apply (Phase 1c §5.4 R1 / §4.7 deletion of the "
        "bespoke VAE-PP call site)."
    )
    assert orch_apply.call_count == 1, (
        "ON path: Orchestrator.apply MUST be called exactly once when "
        "use_init_dispatch=True (Phase 1c §5.4 R1 flag gate)."
    )
    assert legacy_sp.call_count == 0, (
        "ON path: legacy _apply_sequence_parallel_if_enabled MUST NOT be "
        "called; SP is reached via Orchestrator.apply alongside VAE-PP."
    )

    # The dispatched plan MUST contain a VaePatchParallelStrategy carrying
    # the active vae_patch_parallel_size so the orchestrator's APPLY_ORDER
    # iteration runs vae_pp.apply with the configured degree (§4.1.2).
    args, _kwargs = orch_apply.call_args
    plan_arg = args[1]
    vae_pp_modules = [m for m in plan_arg if m.axis == "vae_pp"]
    assert len(vae_pp_modules) == 1, (
        f"ON path: Orchestrator.apply MUST receive exactly one vae_pp "
        f"module when vae_patch_parallel_size=2; got "
        f"{[m.axis for m in plan_arg]}."
    )
    assert isinstance(vae_pp_modules[0], VaePatchParallelStrategy)
    assert vae_pp_modules[0]._degree == 2, (
        "ON path: dispatched VaePatchParallelStrategy must carry the "
        "configured vae_patch_parallel_size (§4.1.2 lower_from_runtime_kwargs)."
    )
