# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T7 (Phase 1c §6.2.1): VAE-PP dispatch ON-vs-OFF GPU equivalence gate.

Server-only / 2-GPU. Runs the SAME ``Tongyi-MAI/Z-Image-Turbo`` recipe
twice — once with ``use_init_dispatch=False`` (legacy bespoke VAE-PP call
site) and once with ``use_init_dispatch=True`` (Orchestrator.apply
dispatch path) — and asserts the decoded image tensors are SHA256
byte-identical. This is the GPU-side equivalence guarantee that backs the
§4.7 "VAE-PP module body unchanged" claim and the §4.8 / G5 "byte-identical
to Phase 1b" coexistence contract.

Recipe (per §6.2.1, copy of the Phase-1 VAE-PP pilot):
* Checkpoint: ``Tongyi-MAI/Z-Image-Turbo`` (HuggingFace).
* Hardware:   2 × GPU (CUDA / ROCm). Skipped on NPU and on hosts with
              fewer than 2 visible devices.
* Config:     ``vae_patch_parallel_size=2``, ``tensor_parallel_size=1``,
              ``cfg_parallel_size=1``, ``ulysses_degree=1``,
              ``ring_degree=1``, ``vae_use_tiling=True`` (the VAE-PP path
              auto-enables it on >=2 ranks; we set it explicitly so the
              flag-OFF baseline observes the same memory-opt knob).
* Sampling:   ``seed=42``, ``num_inference_steps=4``,
              ``height=width=256``, prompt
              ``"a quiet mountain lake at dawn"``.

Runnable invocation (project lead pastes this on the assigned GPU server
once Phase 1c lands; per ``server-workflow.mdc`` execution stays on the
server, code editing stays local). Per §6.2.1 / the benchmarking rule the
first run is warm-up and discarded; this gate compares OFF vs ON only on
their warm runs (the first ``OmniRunner.generate(...)`` in each invocation
is intrinsically warm-up because each ``OmniRunner`` constructs a fresh
engine; we drive it with two prompts and read the SECOND output)::

    CUDA_VISIBLE_DEVICES=0,1 \\
    PYTHONHASHSEED=0 \\
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
    TORCH_USE_DETERMINISTIC_ALGORITHMS=1 \\
    .venv/bin/python -m pytest \\
      tests/diffusion/distributed/test_vae_pp_dispatch_equivalence.py::test_byte_identical_off_vs_on \\
      -s -v

Pass criterion: ``sha_off == sha_on`` (byte-identical SHA256). Any drift
fails the gate and blocks Phase 1c landing.

This file is intentionally distinct from
``tests/diffusion/distributed/test_distributed_vae_executor.py`` (a CPU
unit test of the distributed-VAE executor wrapper, NOT a gate) and
``tests/diffusion/distributed/test_zimage_parallelism.py`` (which already
exercises VAE-PP correctness against a non-VAE-PP baseline but does not
toggle ``use_init_dispatch``). Adding a ``use_init_dispatch``
parameterization to ``test_zimage_parallelism.py`` would entangle this
new gate with the unrelated TP / VAE-PP correctness matrix, so §6.2.1
elects a dedicated file.
"""

from __future__ import annotations

import gc
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# Mirror the spawn-method requirement that ``test_zimage_parallelism.py``
# pins (the worker-multiproc path on this harness needs spawn for CUDA
# initialization to be deterministic across processes).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Recipe constants — frozen by §6.2.1.
# ---------------------------------------------------------------------------
PROMPT = "a quiet mountain lake at dawn"
HEIGHT = 256
WIDTH = 256
NUM_INFERENCE_STEPS = 4
SEED = 42
VAE_PATCH_PARALLEL_SIZE = 2


def _get_zimage_model() -> str:
    """Match ``test_zimage_parallelism.py``'s override hook for offline runs."""
    return os.environ.get("VLLM_TEST_ZIMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")


def _cleanup_distributed() -> None:
    """Match ``test_sequence_parallel.py``'s teardown so back-to-back
    OmniRunner constructions do not collide on stale process groups /
    rank env vars. Required between OFF and ON runs.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
    for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        os.environ.pop(key, None)
    gc.collect()
    if current_omni_platform.is_available():
        current_omni_platform.empty_cache()
        current_omni_platform.synchronize()
    time.sleep(5)


def _extract_single_image(outputs) -> Image.Image:
    """Mirror ``test_zimage_parallelism._extract_single_image``.

    Pulls the single PIL image out of the OmniRequestOutput envelope.
    The byte-hash is computed AFTER this extraction so OFF and ON are
    compared on the same observable surface (the post-processed RGB
    image, which is the user-visible output of the diffusion pipeline).
    """
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise ValueError("Invalid request_output structure or missing 'images' key")

    images = req_out.images
    if images is None or len(images) != 1:
        raise ValueError(f"Expected 1 image, got {0 if images is None else len(images)}")
    return images[0]


def _image_sha256(img: Image.Image) -> str:
    """Stable byte hash of an image's RGB pixels (the §6.2.1 pass criterion).

    Decoded image is first cast to a contiguous ``uint8`` RGB array so the
    hash is deterministic across PIL versions / array memory layouts.
    """
    arr = np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _run_and_sha(*, use_init_dispatch: bool) -> str:
    """Run the §6.2.1 recipe once, return SHA256 of the decoded image.

    Per the benchmarking rule and the §6.2.1 docstring: each ``OmniRunner``
    instance bears intrinsic warm-up overhead (engine boot, weights load,
    cuda graph compile). To keep this gate honest we drive ``generate(...)``
    with TWO prompts and hash the SECOND output — the first is the
    discarded warm-up run.
    """
    parallel_config = DiffusionParallelConfig(
        tensor_parallel_size=1,
        cfg_parallel_size=1,
        ulysses_degree=1,
        ring_degree=1,
        vae_patch_parallel_size=VAE_PATCH_PARALLEL_SIZE,
        use_init_dispatch=use_init_dispatch,
    )
    try:
        with OmniRunner(
            _get_zimage_model(),
            parallel_config=parallel_config,
            # The VAE-PP path auto-enables ``vae_use_tiling`` on >=2 ranks
            # via ``_wire_distributed_vae`` — set it explicitly so both
            # OFF and ON observe the same memory-opt knob and the SHA256
            # equality is decided ONLY by the dispatch route.
            vae_use_tiling=True,
        ) as runner:
            sampling_params = OmniDiffusionSamplingParams(
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=0.0,
                generator=torch.Generator(
                    current_omni_platform.device_type
                ).manual_seed(SEED),
                num_outputs_per_prompt=1,
            )
            # First call = discarded warm-up (per server-workflow rule).
            _ = runner.omni.generate(PROMPT, sampling_params)

            # Re-seed the generator so the timed run's RNG state matches
            # a fresh ``manual_seed(SEED)`` invocation (mirrors the
            # ``test_sequence_parallel._run_inference`` "warmup + timed"
            # idiom). Without this the warm-up call would have advanced
            # the generator state.
            sampling_params_timed = OmniDiffusionSamplingParams(
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=0.0,
                generator=torch.Generator(
                    current_omni_platform.device_type
                ).manual_seed(SEED),
                num_outputs_per_prompt=1,
            )
            outputs = runner.omni.generate(PROMPT, sampling_params_timed)
            img = _extract_single_image(outputs)
            return _image_sha256(img)
    finally:
        _cleanup_distributed()


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 2, "rocm": 2})
def test_byte_identical_off_vs_on() -> None:
    """OFF vs ON of ``use_init_dispatch`` MUST produce byte-identical output.

    Pass criterion (§6.2.1): ``sha_off == sha_on``. Any drift indicates
    the dispatch path is NOT equivalent to the bespoke inline call site
    and blocks Phase 1c landing.

    Skips:
    * NPU — Z-Image VAE-PP gate is CUDA / ROCm only today (mirrors
      ``test_zimage_parallelism.test_zimage_vae_patch_parallel_tp2``).
    * Fewer than 2 visible devices — ``vae_patch_parallel_size=2``
      requires 2 GPUs.

    For the runnable invocation (env vars + pytest line) see this file's
    module docstring.
    """
    if current_omni_platform.is_npu():
        pytest.skip(
            "VAE-PP dispatch equivalence gate is CUDA / ROCm only "
            "(matches Phase-1 Z-Image VAE-PP gate scope)."
        )
    if (
        not current_omni_platform.is_available()
        or current_omni_platform.device_count() < VAE_PATCH_PARALLEL_SIZE
    ):
        pytest.skip(
            f"Requires {VAE_PATCH_PARALLEL_SIZE} GPUs, "
            f"found {current_omni_platform.device_count() if current_omni_platform.is_available() else 0}."
        )

    print("\n" + "=" * 70)
    print(
        f"VAE-PP dispatch equivalence gate — Model: {_get_zimage_model()} "
        f"(vae_patch_parallel_size={VAE_PATCH_PARALLEL_SIZE})"
    )
    print("=" * 70)

    sha_off = _run_and_sha(use_init_dispatch=False)
    print(f"[OFF / bespoke VAE-PP call]   sha256={sha_off}")

    sha_on = _run_and_sha(use_init_dispatch=True)
    print(f"[ON  / Orchestrator.apply ]   sha256={sha_on}")

    assert sha_off == sha_on, (
        f"VAE-PP dispatch equivalence broke (Phase 1c §6.2.1 / G5): "
        f"OFF={sha_off} ON={sha_on}. The init-dispatch path is NOT "
        f"byte-identical to the bespoke inline call site for "
        f"{_get_zimage_model()} at vae_patch_parallel_size="
        f"{VAE_PATCH_PARALLEL_SIZE}."
    )
