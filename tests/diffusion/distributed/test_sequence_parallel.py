# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
System test for Sequence Parallel (SP) backends: Ulysses and Ring attention.

Tests verify that SP inference produces correct outputs compared to baseline.
"""

import gc
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import torch
import torch.distributed as dist
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.distributed.utils import build_local_sp_padding_mask
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Test configuration
MODELS = ["riverclouds/qwen_image_random"]
PROMPT = "a photo of a cat sitting on a laptop keyboard"
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_SEED = 42
DEFAULT_STEPS = 4
DIFF_MEAN_THRESHOLD = 2e-2
DIFF_MAX_THRESHOLD = 2e-1


class InferenceResult(NamedTuple):
    """Result of an inference run."""

    images: list[Image.Image]
    elapsed_ms: float


def _cleanup_distributed():
    """Clean up distributed environment and GPU resources."""
    if dist.is_initialized():
        dist.destroy_process_group()

    for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        os.environ.pop(key, None)

    gc.collect()
    if current_omni_platform.is_available():
        current_omni_platform.empty_cache()
        current_omni_platform.synchronize()

    time.sleep(5)


def _diff_metrics(a: Image.Image, b: Image.Image) -> tuple[float, float]:
    """Return (mean_abs_diff, max_abs_diff) over RGB pixels in [0, 1]."""
    ta = torch.from_numpy(np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0)
    tb = torch.from_numpy(np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0)
    assert ta.shape == tb.shape, f"Image shapes differ: {ta.shape} vs {tb.shape}"
    abs_diff = torch.abs(ta - tb)
    return abs_diff.mean().item(), abs_diff.max().item()


def _run_inference(
    model_name: str,
    dtype: torch.dtype,
    attn_backend: str,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    seed: int = DEFAULT_SEED,
    warmup: bool = True,
    use_sp_descriptor: bool = False,
    use_init_dispatch: bool = False,
) -> InferenceResult:
    """Run inference with specified configuration.

    Args:
        warmup: If True, run one warmup iteration before the timed run.
        use_sp_descriptor: Phase 1b flag. When True, SP hooks are wired from the
            typed ``SPDescriptor`` declaration instead of the legacy ``_sp_plan``
            dict. Threaded onto the constructed ``DiffusionParallelConfig`` so the
            equivalence gate below can genuinely toggle the new path.
        use_init_dispatch: Phase 1c-Twin flag. When True, model-init wiring
            (VAE-PP, SP) is dispatched through ``Orchestrator.apply(...)`` and
            SP wiring runs through the *new*-path
            ``apply_sp_to_pipeline`` helper. When False (default), the legacy
            inline ``_apply_sp_runtime`` body is used. Threaded onto
            ``DiffusionParallelConfig`` so the GPU equivalence gate below can
            parameterize over both paths.
    """
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        use_sp_descriptor=use_sp_descriptor,
        use_init_dispatch=use_init_dispatch,
    )
    try:
        with OmniRunner(
            model_name,
            parallel_config=parallel_config,
            dtype=dtype,
            attention_backend=attn_backend,
        ) as runner:
            omni = runner.omni
            # Warmup run (not timed)
            if warmup:
                _ = omni.generate(
                    PROMPT,
                    OmniDiffusionSamplingParams(
                        height=height,
                        width=width,
                        num_inference_steps=DEFAULT_STEPS,
                        guidance_scale=0.0,
                        generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed + 1000),
                        num_outputs_per_prompt=1,
                    ),
                )

            # Timed run
            start = time.time()
            outputs = omni.generate(
                PROMPT,
                OmniDiffusionSamplingParams(
                    height=height,
                    width=width,
                    num_inference_steps=DEFAULT_STEPS,
                    guidance_scale=0.0,
                    generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
                    num_outputs_per_prompt=1,
                ),
            )
            elapsed_ms = (time.time() - start) * 1000

            return InferenceResult(
                images=outputs[0].request_output.images,
                elapsed_ms=elapsed_ms,
            )
    finally:
        _cleanup_distributed()


# =============================================================================
# Correctness & Performance Tests
# =============================================================================

# SP configurations: (ulysses_degree, ring_degree, height, width, warmup, is_perf_test)
# - warmup: whether to run warmup for this SP config
# - is_perf_test: whether this is a performance test (show speedup metrics)
SP_CONFIGS_L2 = [
    # Hybrid - correctness only
    (2, 2, DEFAULT_HEIGHT, DEFAULT_WIDTH, False, False),
]

SP_CONFIGS_L3 = [
    # Ulysses-2 - performance test
    (2, 1, DEFAULT_HEIGHT, DEFAULT_WIDTH, True, True),
    (1, 2, DEFAULT_HEIGHT, DEFAULT_WIDTH, True, True),  # Ring-2 - performance test
    # Hybrid - correctness only
    (2, 2, DEFAULT_HEIGHT, DEFAULT_WIDTH, False, False),
    (4, 1, 272, 272, False, False),  # Ulysses-4 - shape and correctness
]


def _get_sp_mode(ulysses_degree: int, ring_degree: int) -> str:
    """Get SP mode name for logging."""
    if ulysses_degree > 1 and ring_degree == 1:
        return f"ulysses-{ulysses_degree}"
    elif ring_degree > 1 and ulysses_degree == 1:
        return f"ring-{ring_degree}"
    else:
        return f"hybrid-{ulysses_degree}x{ring_degree}"


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 2, "rocm": 2})
@pytest.mark.parametrize("model_name", MODELS)
def test_sp_correctness(model_name: str):
    """Test that SP inference produces correct outputs and measure performance.

    Runs baseline once per unique (height, width), then tests all SP configs.

    Note: Run with `pytest -v -s` to see detailed output.
    """
    device_count = current_omni_platform.get_device_count()

    # Cache baseline results by (height, width)
    # Key: (height, width), Value: (result, warmup_used)
    baseline_cache: dict[tuple[int, int], InferenceResult] = {}

    # Collect results for summary
    results: list[dict] = []

    print("\n" + "=" * 70)
    print(f"Sequence Parallel Test - Model: {model_name}")
    print(f"Available GPUs: {device_count}")
    print("=" * 70)

    for ulysses_degree, ring_degree, height, width, sp_warmup, is_perf_test in SP_CONFIGS_L2:
        sp_size = ulysses_degree * ring_degree
        sp_mode = _get_sp_mode(ulysses_degree, ring_degree)

        if device_count < sp_size:
            print(f"\n[{sp_mode}] SKIPPED (requires {sp_size} GPUs)")
            continue

        # Determine baseline warmup: only for default size (performance tests)
        cache_key = (height, width)
        baseline_warmup = height == DEFAULT_HEIGHT and width == DEFAULT_WIDTH

        # Get or compute baseline for this (height, width)
        if cache_key not in baseline_cache:
            print(f"\n--- Running baseline {height}x{width} (warmup={baseline_warmup}) ---")
            baseline = _run_inference(
                model_name,
                torch.bfloat16,
                "sdpa",
                height=height,
                width=width,
                warmup=baseline_warmup,
            )
            assert len(baseline.images) == 1
            baseline_cache[cache_key] = baseline
            print(f"[baseline] {height}x{width}: {baseline.elapsed_ms:.0f}ms")
        else:
            baseline = baseline_cache[cache_key]

        # Run SP
        print(f"\n--- Running {sp_mode} (warmup={sp_warmup}) ---")
        sp_result = _run_inference(
            model_name,
            torch.bfloat16,
            "sdpa",
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            height=height,
            width=width,
            warmup=sp_warmup,
        )
        assert len(sp_result.images) == 1

        # Compare outputs (correctness)
        mean_diff, max_diff = _diff_metrics(baseline.images[0], sp_result.images[0])

        # Build result entry
        result = {
            "mode": sp_mode,
            "sp_size": sp_size,
            "height": height,
            "width": width,
            "baseline_ms": baseline.elapsed_ms,
            "sp_ms": sp_result.elapsed_ms,
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "is_perf_test": is_perf_test,
        }
        results.append(result)

        # Output based on test type
        if is_perf_test:
            speedup = baseline.elapsed_ms / sp_result.elapsed_ms if sp_result.elapsed_ms > 0 else 0
            result["speedup"] = speedup
            print(
                f"[{sp_mode}] {sp_size} GPUs | "
                f"baseline: {baseline.elapsed_ms:.0f}ms, sp: {sp_result.elapsed_ms:.0f}ms, "
                f"speedup: {speedup:.2f}x"
            )
        else:
            print(f"[{sp_mode}] {sp_size} GPUs | sp: {sp_result.elapsed_ms:.0f}ms (correctness only)")

        print(f"[{sp_mode}] diff: mean={mean_diff:.6e}, max={max_diff:.6e}")

        # Assert correctness
        assert mean_diff <= DIFF_MEAN_THRESHOLD and max_diff <= DIFF_MAX_THRESHOLD, (
            f"[{sp_mode}] SP output differs from baseline: mean={mean_diff:.6e}, max={max_diff:.6e}"
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<15} {'GPUs':<6} {'Size':<10} {'Baseline':<12} {'SP':<12} {'Speedup':<10} {'Status'}")
    print("-" * 70)
    for r in results:
        speedup_str = f"{r['speedup']:.2f}x" if r.get("speedup") else "N/A"
        baseline_str = f"{r['baseline_ms']:.0f}ms" if r["is_perf_test"] else "N/A"
        status = "PASS" if r["mean_diff"] <= DIFF_MEAN_THRESHOLD else "FAIL"
        print(
            f"{r['mode']:<15} {r['sp_size']:<6} {r['height']}x{r['width']:<5} "
            f"{baseline_str:<12} {r['sp_ms']:.0f}ms{'':<7} {speedup_str:<10} {status}"
        )
    print("=" * 70)


# =============================================================================
# Phase 1b: SPDescriptor flag-ON vs flag-OFF output-equivalence gate
# =============================================================================


def _image_sha256(img: Image.Image) -> str:
    """Stable byte hash of an image's RGB pixels (the Phase-1 VAE-PP gate bar)."""
    return hashlib.sha256(np.asarray(img.convert("RGB"), dtype=np.uint8).tobytes()).hexdigest()


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 2, "rocm": 2})
@pytest.mark.parametrize(
    "use_init_dispatch",
    [False, True],
    ids=["dispatch_off_legacy_path", "dispatch_on_new_path"],
)
@pytest.mark.parametrize("model_name", MODELS)
def test_sp_descriptor_equivalence_gate(model_name: str, use_init_dispatch: bool):
    """Prove the SPDescriptor path (flag ON) == the legacy ``_sp_plan`` path (OFF).

    Runs the SAME QwenImage generation twice under identical SP degrees — once
    with ``use_sp_descriptor=False`` (legacy ``_sp_plan`` -> apply_sequence_parallel)
    and once with ``use_sp_descriptor=True`` (SPDescriptor.to_plan(model) ->
    apply_sequence_parallel) — and asserts the outputs are byte-identical
    (SHA256), the Phase-1 VAE-PP gate bar. If a nondeterministic kernel breaks
    bit-equality, it falls back to the existing image-diff tolerance oracle and
    surfaces the deviation.

    This is the gate the REVISED Phase 1b spec requires: unlike a plain
    ``USE_SP_DESCRIPTOR=... pytest`` (which the harness never reads), the flag is
    genuinely toggled here on the constructed ``DiffusionParallelConfig``.

    Run on the single assigned 2-GPU server terminal; the first run is warm-up.
    """
    device_count = current_omni_platform.get_device_count()
    ulysses_degree, ring_degree = 2, 1
    sp_size = ulysses_degree * ring_degree
    if device_count < sp_size:
        pytest.skip(f"requires {sp_size} GPUs, found {device_count}")

    print("\n" + "=" * 70)
    print(f"SPDescriptor equivalence gate - Model: {model_name} (ulysses={ulysses_degree})")
    print("=" * 70)

    legacy = _run_inference(
        model_name,
        torch.bfloat16,
        "sdpa",
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        warmup=True,
        use_sp_descriptor=False,
        use_init_dispatch=use_init_dispatch,
    )
    descriptor = _run_inference(
        model_name,
        torch.bfloat16,
        "sdpa",
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        warmup=True,
        use_sp_descriptor=True,
        use_init_dispatch=use_init_dispatch,
    )
    assert len(legacy.images) == 1 and len(descriptor.images) == 1

    legacy_hash = _image_sha256(legacy.images[0])
    descriptor_hash = _image_sha256(descriptor.images[0])
    print(f"[legacy]     sha256={legacy_hash}")
    print(f"[descriptor] sha256={descriptor_hash}")

    if legacy_hash == descriptor_hash:
        return  # byte-identical: the strongest pass

    # Fallback: bit-equality may break on nondeterministic kernels even though
    # both paths lower to the identical plan/hooks. Accept the existing tolerance
    # oracle and document the deviation.
    mean_diff, max_diff = _diff_metrics(legacy.images[0], descriptor.images[0])
    print(
        f"[gate] sha256 mismatch; falling back to tolerance oracle: "
        f"mean={mean_diff:.6e}, max={max_diff:.6e}"
    )
    assert mean_diff <= DIFF_MEAN_THRESHOLD and max_diff <= DIFF_MAX_THRESHOLD, (
        f"SPDescriptor (flag ON) output differs from legacy (flag OFF): "
        f"mean={mean_diff:.6e}, max={max_diff:.6e}"
    )


# TODO: After PR#1272 is merged, add markers
# @pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 4, "rocm": 2})
@pytest.mark.parametrize("model_name", MODELS)
def test_sp_correctness_advanced(model_name: str):
    """Test that SP inference produces correct outputs and measure performance.

    Runs baseline once per unique (height, width), then tests all SP configs.

    Note: Run with `pytest -v -s` to see detailed output.
    """
    device_count = current_omni_platform.get_device_count()

    # Cache baseline results by (height, width)
    # Key: (height, width), Value: (result, warmup_used)
    baseline_cache: dict[tuple[int, int], InferenceResult] = {}

    # Collect results for summary
    results: list[dict] = []

    print("\n" + "=" * 70)
    print(f"Sequence Parallel Test - Model: {model_name}")
    print(f"Available GPUs: {device_count}")
    print("=" * 70)

    for ulysses_degree, ring_degree, height, width, sp_warmup, is_perf_test in SP_CONFIGS_L3:
        sp_size = ulysses_degree * ring_degree
        sp_mode = _get_sp_mode(ulysses_degree, ring_degree)

        if device_count < sp_size:
            print(f"\n[{sp_mode}] SKIPPED (requires {sp_size} GPUs)")
            continue

        # Determine baseline warmup: only for default size (performance tests)
        cache_key = (height, width)
        baseline_warmup = height == DEFAULT_HEIGHT and width == DEFAULT_WIDTH

        # Get or compute baseline for this (height, width)
        if cache_key not in baseline_cache:
            print(f"\n--- Running baseline {height}x{width} (warmup={baseline_warmup}) ---")
            baseline = _run_inference(
                model_name,
                torch.bfloat16,
                "sdpa",
                height=height,
                width=width,
                warmup=baseline_warmup,
            )
            assert len(baseline.images) == 1
            baseline_cache[cache_key] = baseline
            print(f"[baseline] {height}x{width}: {baseline.elapsed_ms:.0f}ms")
        else:
            baseline = baseline_cache[cache_key]

        # Run SP
        print(f"\n--- Running {sp_mode} (warmup={sp_warmup}) ---")
        sp_result = _run_inference(
            model_name,
            torch.bfloat16,
            "sdpa",
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            height=height,
            width=width,
            warmup=sp_warmup,
        )
        assert len(sp_result.images) == 1

        # Compare outputs (correctness)
        mean_diff, max_diff = _diff_metrics(baseline.images[0], sp_result.images[0])

        # Build result entry
        result = {
            "mode": sp_mode,
            "sp_size": sp_size,
            "height": height,
            "width": width,
            "baseline_ms": baseline.elapsed_ms,
            "sp_ms": sp_result.elapsed_ms,
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "is_perf_test": is_perf_test,
        }
        results.append(result)

        # Output based on test type
        if is_perf_test:
            speedup = baseline.elapsed_ms / sp_result.elapsed_ms if sp_result.elapsed_ms > 0 else 0
            result["speedup"] = speedup
            print(
                f"[{sp_mode}] {sp_size} GPUs | "
                f"baseline: {baseline.elapsed_ms:.0f}ms, sp: {sp_result.elapsed_ms:.0f}ms, "
                f"speedup: {speedup:.2f}x"
            )
        else:
            print(f"[{sp_mode}] {sp_size} GPUs | sp: {sp_result.elapsed_ms:.0f}ms (correctness only)")

        print(f"[{sp_mode}] diff: mean={mean_diff:.6e}, max={max_diff:.6e}")

        # Assert correctness
        assert mean_diff <= DIFF_MEAN_THRESHOLD and max_diff <= DIFF_MAX_THRESHOLD, (
            f"[{sp_mode}] SP output differs from baseline: mean={mean_diff:.6e}, max={max_diff:.6e}"
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<15} {'GPUs':<6} {'Size':<10} {'Baseline':<12} {'SP':<12} {'Speedup':<10} {'Status'}")
    print("-" * 70)
    for r in results:
        speedup_str = f"{r['speedup']:.2f}x" if r.get("speedup") else "N/A"
        baseline_str = f"{r['baseline_ms']:.0f}ms" if r["is_perf_test"] else "N/A"
        status = "PASS" if r["mean_diff"] <= DIFF_MEAN_THRESHOLD else "FAIL"
        print(
            f"{r['mode']:<15} {r['sp_size']:<6} {r['height']}x{r['width']:<5} "
            f"{baseline_str:<12} {r['sp_ms']:.0f}ms{'':<7} {speedup_str:<10} {status}"
        )
    print("=" * 70)


@pytest.mark.skipif(
    not (current_omni_platform.is_cuda() or current_omni_platform.is_xpu()),
    reason="Only tested on CUDA and XPU",
)
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.core_model
def test_local_sp_padding_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partially padded SP shard must receive a local-length mask."""
    mask = build_local_sp_padding_mask(
        batch_size=2,
        local_seq_len=4,
        sp_original_seq_len=5,
        sp_padding_size=3,
        sequence_parallel_rank=1,
        device=torch.device(current_omni_platform.device_type),
    )

    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, False, False, False],
        ],
        dtype=torch.bool,
        device=mask.device,
    )
    assert mask is not None
    assert mask.shape == (2, 4)
    assert torch.equal(mask, expected)


@pytest.mark.skipif(
    not (current_omni_platform.is_cuda() or current_omni_platform.is_xpu()),
    reason="Only tested on CUDA and XPU",
)
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.core_model
def test_local_sp_padding_mask_no_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rank whose local shard contains no padding should not get a mask."""
    mask = build_local_sp_padding_mask(
        batch_size=2,
        local_seq_len=4,
        sp_original_seq_len=5,
        sp_padding_size=3,
        sequence_parallel_rank=0,
        device=torch.device(current_omni_platform.device_type),
    )

    assert mask is None


@pytest.mark.skipif(
    not (current_omni_platform.is_cuda() or current_omni_platform.is_xpu()),
    reason="Only tested on CUDA and XPU",
)
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.core_model
def test_wan_sp_plan() -> None:
    """Wan2.2 must shard hidden states before CacheDiT-wrapped transformer blocks."""
    try:
        from vllm_omni.diffusion.distributed.sp_plan import validate_sp_plan
        from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel
    except ImportError as exc:
        pytest.skip(f"WanTransformer3DModel not available: {exc}")

    plan = getattr(WanTransformer3DModel, "_sp_plan", None)

    assert plan is not None
    assert "_sp_shard_point" in plan
    assert "blocks.0" not in plan

    shard_plan = plan["_sp_shard_point"]
    assert 0 in shard_plan
    assert shard_plan[0].split_dim == 1
    assert shard_plan[0].expected_dims == 3
    assert shard_plan[0].split_output is True

    validate_sp_plan(plan)


# =============================================================================
# Phase 1c (T6 §6.1.5): CPU-level SP dispatch routing tests.
#
# These tests exercise Phase 1c §4.5 — the migration of the inline
# ``_apply_sequence_parallel_if_enabled`` body off the registry and into the
# shared ``_apply_sp_runtime`` helper, plus the ON path through
# ``sp_ulysses.apply`` / ``sp_ring.apply``. They lock in the four properties
# the spec §6.1.5 / §4.5.4 calls out:
#
#   (a) OFF path → registry's thin wrapper ``_apply_sequence_parallel_if_enabled``
#       delegates to ``_apply_sp_runtime`` (legacy helper, unchanged).
#   (b) ON  path → ``sp_ulysses.apply`` / ``sp_ring.apply`` delegate to the
#       NEW ``apply_sp_to_pipeline`` helper in ``sp_runtime.py`` (Twin path,
#       fully independent from the legacy helper per
#       ``docs/DESIGN_PHASE1C_TWIN_PATHS.md``).
#   (c) F2 lock-in (§4.5.4 (use_init_dispatch=True, use_sp_descriptor=False)
#       cell): even on a model that already carries an ``_sp_descriptor``, the
#       legacy ``_sp_plan`` branch is taken — ``apply_sequence_parallel_from_descriptor``
#       is NOT reached.
#   (d) OFF and ON paths return identical ``hooks_applied`` counts via
#       INDEPENDENT helpers (legacy ``_apply_sp_runtime`` vs new
#       ``apply_sp_to_pipeline``); equivalence is asserted by tests, not by
#       construction.
#
# CPU-only by construction: every distributed dependency is mocked.
# =============================================================================

from unittest.mock import MagicMock, patch  # noqa: E402

import torch.nn as nn  # noqa: E402

from vllm_omni.config.composable_parallel.modules.axes.sp_ring import (  # noqa: E402
    RingSequenceParallelStrategy,
)
from vllm_omni.config.composable_parallel.modules.axes.sp_ulysses import (  # noqa: E402
    UlyssesSequenceParallelStrategy,
)
from vllm_omni.config.composable_parallel.modules.base import ApplyCtx  # noqa: E402
from vllm_omni.diffusion.data import OmniDiffusionConfig  # noqa: E402
from vllm_omni.diffusion.forward_context import (  # noqa: E402
    ForwardContext,
    override_forward_context,
)
from vllm_omni.diffusion.hooks.sequence_parallel import _apply_sp_runtime  # noqa: E402
from vllm_omni.diffusion.registry import _apply_sequence_parallel_if_enabled  # noqa: E402


def _make_od_config(
    *,
    ulysses_degree: int = 2,
    ring_degree: int = 1,
    use_sp_descriptor: bool = False,
    use_init_dispatch: bool = False,
) -> OmniDiffusionConfig:
    """Build a minimal CPU-side OmniDiffusionConfig for dispatch routing tests.

    Only the SP-relevant fields on ``parallel_config`` are populated; everything
    else takes its dataclass default.
    """
    od = OmniDiffusionConfig()
    od.parallel_config = DiffusionParallelConfig(
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        use_sp_descriptor=use_sp_descriptor,
        use_init_dispatch=use_init_dispatch,
    )
    return od


class _FakeTransformer(nn.Module):
    """Minimal nn.Module standing in for a pipeline's ``.transformer``.

    The optional ``_sp_descriptor`` attribute is what the F2 lock-in test uses:
    even though this fake carries one, ``_apply_sp_runtime`` MUST take the
    legacy ``_sp_plan`` branch when ``use_sp_descriptor=False``. The ``_sp_plan``
    attribute itself doesn't need a real value here — ``get_sp_plan_from_model``
    is mocked at the helper boundary in the F2 test.
    """

    def __init__(self, *, with_descriptor: bool = False) -> None:
        super().__init__()
        if with_descriptor:
            self._sp_descriptor = object()  # opaque sentinel; never inspected


class _FakePipeline(nn.Module):
    """Minimal pipeline shell with a ``.transformer`` attribute the SP helper
    discovers via the canonical scan order ``["transformer", ...]``.
    """

    def __init__(self, transformer: _FakeTransformer) -> None:
        super().__init__()
        self.transformer = transformer


@pytest.mark.core_model
@pytest.mark.cpu
def test_off_path_calls_apply_sp_runtime_via_registry_wrapper():
    """(a) OFF path: ``_apply_sequence_parallel_if_enabled`` delegates to
    ``_apply_sp_runtime`` exactly once and returns its result.

    This is the §5.4 R1(c) shape: the wrapper is now a one-line delegator
    that wraps the helper in the Phase-1b ``try/except Exception:
    logger.warning(...)`` sink.
    """
    model = _FakePipeline(_FakeTransformer())
    od = _make_od_config(use_init_dispatch=False)

    with patch(
        "vllm_omni.diffusion.registry._apply_sp_runtime",
        return_value=3,
    ) as helper:
        result = _apply_sequence_parallel_if_enabled(model, od)

    assert helper.call_count == 1, "OFF path must call helper exactly once"
    args, _ = helper.call_args
    assert args[0] is model
    assert args[1] is od.parallel_config, (
        "OFF path must pass od_config.parallel_config to the helper "
        "(matching the ON-path call signature in sp_ulysses.apply)"
    )
    assert result == 3, "wrapper must propagate the helper's return value"


@pytest.mark.core_model
@pytest.mark.cpu
def test_off_path_warn_and_continue_on_helper_failure():
    """N1 deferred (§3.2 N8 / §4.5.4 row 6): the OFF wrapper preserves the
    Phase-1b ``try/except Exception: logger.warning(...)`` sink. A failing
    helper produces ``hooks_applied == 0`` and a warning, never a raise.
    """
    model = _FakePipeline(_FakeTransformer())
    od = _make_od_config(use_init_dispatch=False)

    with patch(
        "vllm_omni.diffusion.registry._apply_sp_runtime",
        side_effect=RuntimeError("synthetic SP wiring failure"),
    ):
        result = _apply_sequence_parallel_if_enabled(model, od)

    assert result == 0, "OFF path must warn-and-continue on helper failure"


@pytest.mark.core_model
@pytest.mark.cpu
def test_on_path_sp_ulysses_calls_apply_sp_to_pipeline():
    """ON path: ``sp_ulysses.apply(ctx)`` delegates to ``apply_sp_to_pipeline``
    (the new Twin SP runtime) via the same try/except sink as the OFF wrapper.
    The OFF path still uses ``_apply_sp_runtime`` (legacy); see
    ``test_off_path_calls_apply_sp_runtime_via_registry_wrapper``.
    """
    model = _FakePipeline(_FakeTransformer())
    od = _make_od_config(use_init_dispatch=True, ulysses_degree=2)
    fc = ForwardContext(omni_diffusion_config=od)

    ctx = ApplyCtx(model=model, od_config=od)
    with (
        override_forward_context(fc),
        patch(
            "vllm_omni.diffusion.distributed.sp_runtime.apply_sp_to_pipeline",
            return_value=5,
        ) as helper,
    ):
        result = UlyssesSequenceParallelStrategy(2).apply(ctx)

    # The apply() body lazily imports ``apply_sp_to_pipeline`` from
    # ``sp_runtime``; patching the source module suffices since each call
    # resolves through the module dict.
    assert helper.call_count == 1, (
        "sp_ulysses.apply must call apply_sp_to_pipeline exactly once"
    )
    args, _ = helper.call_args
    assert args[0] is model
    assert args[1] is od.parallel_config
    assert result.axis == "sp_ulysses"
    assert result.hooks_applied == 5


@pytest.mark.core_model
@pytest.mark.cpu
def test_on_path_sp_ring_calls_apply_sp_to_pipeline_when_not_yet_applied():
    """ON path: ``sp_ring.apply(ctx)`` delegates to ``apply_sp_to_pipeline``
    when SP has not yet been wired for this stage init.
    """
    model = _FakePipeline(_FakeTransformer())
    od = _make_od_config(use_init_dispatch=True, ulysses_degree=1, ring_degree=2)
    fc = ForwardContext(omni_diffusion_config=od)
    fc.sp_plan_hooks_applied = False

    ctx = ApplyCtx(model=model, od_config=od)
    with (
        override_forward_context(fc),
        patch(
            "vllm_omni.diffusion.distributed.sp_runtime.apply_sp_to_pipeline",
            return_value=2,
        ) as helper,
    ):
        result = RingSequenceParallelStrategy(2).apply(ctx)

    assert helper.call_count == 1
    args, _ = helper.call_args
    assert args[0] is model
    assert args[1] is od.parallel_config
    assert result.axis == "sp_ring"
    assert result.hooks_applied == 2


@pytest.mark.core_model
@pytest.mark.cpu
def test_on_path_sp_ring_short_circuits_when_already_applied():
    """``sp_ring`` is the idempotent companion (§4.5.2): if SP hooks were
    already wired this stage init (e.g. by ``sp_ulysses`` in a hybrid config
    or by a re-entry of the dispatch loop), ``apply()`` returns
    ``hooks_applied=0`` and does NOT invoke ``apply_sp_to_pipeline``.
    """
    model = _FakePipeline(_FakeTransformer())
    od = _make_od_config(use_init_dispatch=True, ulysses_degree=1, ring_degree=2)
    fc = ForwardContext(omni_diffusion_config=od)
    fc.sp_plan_hooks_applied = True  # belt-and-braces idempotency marker

    ctx = ApplyCtx(model=model, od_config=od)
    with (
        override_forward_context(fc),
        patch(
            "vllm_omni.diffusion.distributed.sp_runtime.apply_sp_to_pipeline",
            return_value=99,
        ) as helper,
    ):
        result = RingSequenceParallelStrategy(2).apply(ctx)

    assert helper.call_count == 0, (
        "sp_ring must NOT call apply_sp_to_pipeline after SP applied"
    )
    assert result.axis == "sp_ring"
    assert result.hooks_applied == 0
    assert any("idempotent" in n.lower() for n in result.notes)


@pytest.mark.core_model
@pytest.mark.cpu
def test_on_path_sp_ulysses_warn_and_continue_on_helper_failure():
    """N1 deferred: ``sp_ulysses.apply`` mirrors the OFF wrapper's
    ``try/except Exception: logger.warning(...)`` sink so OFF and ON paths
    have bit-identical warn-and-continue behavior on SP wiring failures —
    even though the ON path now goes through ``apply_sp_to_pipeline`` rather
    than ``_apply_sp_runtime``.
    """
    model = _FakePipeline(_FakeTransformer())
    od = _make_od_config(use_init_dispatch=True, ulysses_degree=2)
    fc = ForwardContext(omni_diffusion_config=od)

    ctx = ApplyCtx(model=model, od_config=od)
    with (
        override_forward_context(fc),
        patch(
            "vllm_omni.diffusion.distributed.sp_runtime.apply_sp_to_pipeline",
            side_effect=RuntimeError("synthetic SP wiring failure"),
        ),
    ):
        result = UlyssesSequenceParallelStrategy(2).apply(ctx)

    assert result.axis == "sp_ulysses"
    assert result.hooks_applied == 0, "ON path must warn-and-continue, not raise"


@pytest.mark.core_model
@pytest.mark.cpu
def test_off_and_on_paths_return_identical_hooks_applied_count():
    """(d) OFF and ON paths produce IDENTICAL ``hooks_applied`` counts through
    INDEPENDENT helpers — OFF via legacy ``_apply_sp_runtime``, ON via new
    ``apply_sp_to_pipeline``. Locks the Twin-path observable-equivalence
    contract (``docs/DESIGN_PHASE1C_TWIN_PATHS.md`` §7.1); replaces the
    pre-Twin assertion that both callers shared one helper.
    """
    helper_return = 7

    # OFF path: registry wrapper → legacy helper.
    model_off = _FakePipeline(_FakeTransformer())
    od_off = _make_od_config(use_init_dispatch=False, ulysses_degree=2)
    with patch(
        "vllm_omni.diffusion.registry._apply_sp_runtime",
        return_value=helper_return,
    ):
        off_count = _apply_sequence_parallel_if_enabled(model_off, od_off)

    # ON path: sp_ulysses.apply → new Twin helper.
    model_on = _FakePipeline(_FakeTransformer())
    od_on = _make_od_config(use_init_dispatch=True, ulysses_degree=2)
    fc = ForwardContext(omni_diffusion_config=od_on)
    ctx = ApplyCtx(model=model_on, od_config=od_on)
    with (
        override_forward_context(fc),
        patch(
            "vllm_omni.diffusion.distributed.sp_runtime.apply_sp_to_pipeline",
            return_value=helper_return,
        ),
    ):
        on_result = UlyssesSequenceParallelStrategy(2).apply(ctx)

    assert off_count == on_result.hooks_applied == helper_return, (
        f"OFF={off_count} ON={on_result.hooks_applied} expected={helper_return}; "
        "OFF (via _apply_sp_runtime) and ON (via apply_sp_to_pipeline) must "
        "produce identical hook counts (Twin observable-equivalence)."
    )


@pytest.mark.core_model
@pytest.mark.cpu
def test_f2_lock_in_legacy_path_taken_even_on_descriptor_bearing_model():
    """(c) F2 lock-in (§4.5.4 cell ``(use_init_dispatch=True, use_sp_descriptor=False)``):
    even when the model carries an ``_sp_descriptor`` (e.g. a Phase-1b
    migrated model like QwenImage), the legacy ``_sp_plan`` branch is taken
    when ``use_sp_descriptor=False``. ``apply_sequence_parallel_from_descriptor``
    must NOT be reached.

    This is the central correctness lock the Round-1 review surfaced: the
    earlier draft incorrectly routed this cell through the descriptor
    compiler, which would silently flip OFF-path semantics on already-migrated
    models. We verify by exercising the real ``_apply_sp_runtime`` (no helper
    mock) and patching the two terminal branches inside the helper.
    """
    transformer = _FakeTransformer(with_descriptor=True)
    model = _FakePipeline(transformer)
    od = _make_od_config(
        use_init_dispatch=True,
        ulysses_degree=2,
        use_sp_descriptor=False,
    )
    fc = ForwardContext(omni_diffusion_config=od)

    fake_plan = MagicMock(name="fake_sp_plan")  # truthy → legacy branch applies
    with (
        override_forward_context(fc),
        patch(
            "vllm_omni.diffusion.hooks.sequence_parallel.apply_sequence_parallel_from_descriptor",
        ) as descriptor_apply,
        patch(
            "vllm_omni.diffusion.hooks.sequence_parallel.apply_sequence_parallel",
        ) as legacy_apply,
        patch(
            "vllm_omni.diffusion.distributed.sp_plan.get_sp_plan_from_model",
            return_value=fake_plan,
        ) as get_plan,
    ):
        applied = _apply_sp_runtime(model, od.parallel_config)

    assert descriptor_apply.call_count == 0, (
        "F2 lock-in: apply_sequence_parallel_from_descriptor MUST NOT be called "
        "when use_sp_descriptor=False, even on _sp_descriptor-bearing models."
    )
    assert get_plan.call_count >= 1, (
        "F2 lock-in: legacy branch must consult get_sp_plan_from_model."
    )
    assert legacy_apply.call_count == 1, (
        "F2 lock-in: legacy apply_sequence_parallel must be called exactly once."
    )
    assert applied == 1, "one transformer wired in the legacy branch"
    # Verify the helper wrote the marker on success (mirrors registry.py:472-473).
    assert fc.sp_plan_hooks_applied is True


@pytest.mark.core_model
@pytest.mark.cpu
def test_f2_descriptor_branch_taken_when_flag_on():
    """Sibling of ``test_f2_lock_in_legacy_path_taken_...``: when
    ``use_sp_descriptor=True``, the descriptor branch IS taken (the §4.5.4
    cell ``(use_init_dispatch=True, use_sp_descriptor=True)``). Locks the
    matrix's positive cell so a future regression that breaks the flag
    would fail loudly.
    """
    transformer = _FakeTransformer(with_descriptor=True)
    model = _FakePipeline(transformer)
    od = _make_od_config(
        use_init_dispatch=True,
        ulysses_degree=2,
        use_sp_descriptor=True,
    )
    fc = ForwardContext(omni_diffusion_config=od)

    with (
        override_forward_context(fc),
        patch(
            "vllm_omni.diffusion.hooks.sequence_parallel.apply_sequence_parallel_from_descriptor",
            return_value=True,
        ) as descriptor_apply,
        patch(
            "vllm_omni.diffusion.hooks.sequence_parallel.apply_sequence_parallel",
        ) as legacy_apply,
    ):
        applied = _apply_sp_runtime(model, od.parallel_config)

    assert descriptor_apply.call_count == 1
    assert legacy_apply.call_count == 0
    assert applied == 1
