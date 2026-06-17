# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end plumbing test for the Phase 1c ``use_init_dispatch`` flag.

This is the F3 anti-regression test for Phase 1c (track T1 / §6.1.4 T4b /
§7 R3 of ``docs/DESIGN_PHASE1C_INIT_DISPATCH.md``). It exercises the same
defect class that ``REVIEW_PHASE1B_SP_IMPL.md`` MUST-FIX caught for
``use_sp_descriptor``: a flag added to ``DiffusionParallelConfig`` that
silently does not reach runtime because one of the user-facing entry points
(engine args / deploy YAML / CLI) was not updated.

The test mirrors the structure of
``tests/entrypoints/test_async_omni_diffusion_config.py`` (where the Phase-1b
``use_sp_descriptor`` plumbing is exercised). It asserts only that the flag
survives the kwargs / YAML / CLI -> ``DiffusionParallelConfig`` pipeline; it
does NOT call ``Orchestrator.apply`` and therefore has no dependency on
Phase 1c tracks T2 / T3 / T4.
"""
from __future__ import annotations

import pytest

from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    _build_engine_args,
    _parse_stage_deploy,
)
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Flat engine kwargs path (the "bare single-stage diffusion fallback" Phase 1b
# MUST-FIX caught for use_sp_descriptor).
# ---------------------------------------------------------------------------


def test_default_stage_config_propagates_use_init_dispatch() -> None:
    """A flat ``use_init_dispatch`` engine arg (CLI / deploy YAML / override)
    must reach ``DiffusionParallelConfig.use_init_dispatch`` in the fallback
    parallel_config construction. Without this, a deploy-YAML/CLI opt-in
    silently stays OFF at runtime — the Phase-1b regression class."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "ulysses_degree": 2,
            "use_init_dispatch": True,
        }
    )[0]

    parallel_config = stage_cfg["engine_args"]["parallel_config"]
    assert parallel_config.use_init_dispatch is True


def test_default_stage_config_use_init_dispatch_defaults_off() -> None:
    """Default remains OFF (byte-identical legacy path) when the flag is unset."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "ulysses_degree": 2,
        }
    )[0]

    parallel_config = stage_cfg["engine_args"]["parallel_config"]
    assert parallel_config.use_init_dispatch is False


def test_default_stage_config_use_init_dispatch_preserves_explicit_false() -> None:
    """An explicit ``False`` must stay OFF (not be coerced to a truthy value)."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "ulysses_degree": 2,
            "use_init_dispatch": False,
        }
    )[0]

    parallel_config = stage_cfg["engine_args"]["parallel_config"]
    assert parallel_config.use_init_dispatch is False


# ---------------------------------------------------------------------------
# CLI surface (--use-init-dispatch).
# ---------------------------------------------------------------------------


def test_serve_cli_accepts_use_init_dispatch() -> None:
    """The diffusion serve CLI exposes ``--use-init-dispatch`` and wires it
    through to ``DiffusionParallelConfig.use_init_dispatch`` (F9 / E5)."""
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "Qwen/Qwen-Image",
            "--omni",
            "--usp",
            "2",
            "--use-init-dispatch",
        ]
    )

    explicit_kwargs = args.get_explicit_kwargs_dict()
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(explicit_kwargs)[0]
    parallel_config = stage_cfg["engine_args"]["parallel_config"]

    assert args.use_init_dispatch is True
    assert parallel_config.use_init_dispatch is True


def test_serve_cli_use_init_dispatch_defaults_off() -> None:
    """Without the flag, the CLI default keeps init-dispatch OFF and does
    NOT clobber a deploy-YAML/stage opt-in by appearing in the explicit-kwargs
    dict."""
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "Qwen/Qwen-Image",
            "--omni",
            "--usp",
            "2",
        ]
    )

    explicit_kwargs = args.get_explicit_kwargs_dict()
    # Not explicitly passed -> not present in the explicit kwargs dict, so a
    # deploy-YAML/stage opt-in is never clobbered by a CLI default.
    assert "use_init_dispatch" not in explicit_kwargs
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(explicit_kwargs)[0]
    assert stage_cfg["engine_args"]["parallel_config"].use_init_dispatch is False


# ---------------------------------------------------------------------------
# Deploy YAML stage path (StageDeployConfig -> _build_engine_args ->
# _create_default_diffusion_stage_cfg). This is the path the spec calls out
# in §6.1.4 T4b and that ``REVIEW_PHASE1B_SP_IMPL.md`` MUST-FIX item proved
# is easy to miss.
# ---------------------------------------------------------------------------


def _minimal_diffusion_pipeline_and_deploy(
    *,
    use_init_dispatch: bool | None,
) -> tuple[StagePipelineConfig, StageDeployConfig, PipelineConfig, DeployConfig]:
    """Build the minimal (ps, ds, pipeline, deploy) tuple needed to exercise
    ``_build_engine_args`` for a single diffusion stage. The deploy stage is
    parsed via ``_parse_stage_deploy`` so the test follows the same code path
    a real deploy YAML would, including the ``engine_args`` flattening loop."""
    stage_data: dict[str, object] = {
        "stage_id": 0,
        "engine_args": {
            "ulysses_degree": 2,
        },
    }
    if use_init_dispatch is not None:
        stage_data["engine_args"]["use_init_dispatch"] = use_init_dispatch  # type: ignore[index]
    ds = _parse_stage_deploy(stage_data)

    ps = StagePipelineConfig(
        stage_id=0,
        model_stage="diffusion",
        execution_type=StageExecutionType.DIFFUSION,
        model_arch="dummy-diffusion-arch",
    )
    pipeline = PipelineConfig(model_type="dummy", model_arch="dummy-diffusion-arch", stages=(ps,))
    deploy = DeployConfig(stages=[ds])
    return ps, ds, pipeline, deploy


def test_stage_deploy_config_carries_use_init_dispatch_field() -> None:
    """``StageDeployConfig`` (E2) declares ``use_init_dispatch: bool | None``
    and ``_parse_stage_deploy`` populates it from a deploy YAML stage."""
    ds = _parse_stage_deploy(
        {
            "stage_id": 0,
            "engine_args": {
                "use_init_dispatch": True,
                "ulysses_degree": 2,
            },
        }
    )
    assert ds.use_init_dispatch is True


def test_deploy_yaml_use_init_dispatch_reaches_parallel_config() -> None:
    """End-to-end deploy YAML path: ``use_init_dispatch: true`` in a stage's
    engine_args parses into ``StageDeployConfig.use_init_dispatch=True``, is
    copied onto the stage's flat engine_args by the existing
    ``_build_engine_args`` mover (stage_config.py:861-866), and ultimately
    lands on ``DiffusionParallelConfig.use_init_dispatch`` when the worker
    reconstructs the parallel config via
    ``_create_default_diffusion_stage_cfg``."""
    ps, ds, pipeline, deploy = _minimal_diffusion_pipeline_and_deploy(
        use_init_dispatch=True,
    )

    engine_args = _build_engine_args(ps, ds, pipeline, deploy, next_stage_proc=None)
    # The existing mover at stage_config.py:861-866 copies non-None override
    # fields onto the stage's yaml_engine_args without any further change.
    assert engine_args["use_init_dispatch"] is True

    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(engine_args)[0]
    parallel_config = stage_cfg["engine_args"]["parallel_config"]
    assert parallel_config.use_init_dispatch is True


def test_deploy_yaml_use_init_dispatch_defaults_off() -> None:
    """A deploy YAML stage that does NOT set ``use_init_dispatch`` keeps the
    Phase-1c flag OFF — byte-identical to pre-1c behavior."""
    ps, ds, pipeline, deploy = _minimal_diffusion_pipeline_and_deploy(
        use_init_dispatch=None,
    )

    engine_args = _build_engine_args(ps, ds, pipeline, deploy, next_stage_proc=None)
    # When the deploy override is None, the mover skips the field (see the
    # `v is None: continue` guard at stage_config.py:863) so the engine_args
    # dict does not carry it. Downstream defaults take over.
    assert "use_init_dispatch" not in engine_args

    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(engine_args)[0]
    parallel_config = stage_cfg["engine_args"]["parallel_config"]
    assert parallel_config.use_init_dispatch is False
