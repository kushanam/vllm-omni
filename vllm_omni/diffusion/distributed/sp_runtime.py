# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Truly independent SP runtime for the new (Phase 1c-Twin) dispatch path.

This module is the *new* SP wiring layer. It is the body that
``UlyssesSequenceParallelStrategy.apply()`` and
``RingSequenceParallelStrategy.apply()`` dispatch to under the Phase-1c
``Orchestrator`` loop. It does NOT call into the legacy SP-wiring helper in
``vllm_omni.diffusion.hooks.sequence_parallel`` — that helper stays frozen
and powers only the legacy OFF path
(``registry._apply_sequence_parallel_if_enabled``).

The only thing this module imports from ``hooks/sequence_parallel.py`` is
the two pure low-level installers
(:func:`apply_sequence_parallel` and
:func:`apply_sequence_parallel_from_descriptor`). Those are intentionally
shared between the twin paths because they are pure torch-distributed
primitives that take a typed plan / descriptor and register
``SequenceParallelSplitHook`` / ``SequenceParallelGatherHook`` instances —
no model-shape policy, no flag dispatch, no ``mode = "hybrid"`` snippet, no
``forward_context`` write.

Failure-mode contract: this module does NOT swallow exceptions. The two
callers (``sp_ulysses.apply()`` / ``sp_ring.apply()``) wrap the call in an
identical ``try/except Exception: logger.warning(...)`` sink so OFF and ON
paths share warn-and-continue semantics on SP wiring failure (N1 still
deferred per ``DESIGN_PHASE1C_INIT_DISPATCH.md`` §3.2 N8).
"""
from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.pipeline_adapters import (
    get_transformers_for_pipeline,
)
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
from vllm_omni.diffusion.hooks.sequence_parallel import (
    apply_sequence_parallel,
    apply_sequence_parallel_from_descriptor,
)

logger = init_logger(__name__)


def _mode_tag(sp_config: SequenceParallelConfig) -> str:
    """Single inlined replacement for the duplicated 3-line snippet at
    ``hooks/sequence_parallel.py:755-759``, ``:778-782``, and ``:950-955``.

    Log-only; no control-flow consumer. PRIVATE to ``sp_runtime.py``: it is
    intentionally not re-exported. The third legacy copy in
    ``enable_sequence_parallel_for_model`` stays frozen until the legacy
    path is retired (Phase 1c-Twin §11).
    """
    if sp_config.ulysses_degree > 1 and sp_config.ring_degree > 1:
        return "hybrid"
    if sp_config.ulysses_degree > 1:
        return "ulysses"
    return "ring"


def apply_sp_to_pipeline(pipeline, parallel_config) -> int:
    """Apply SP wiring to a pipeline. Truly independent twin of the legacy
    SP runtime helper at ``hooks/sequence_parallel.py:652``.

    The NEW path's single entry point. Called by ``sp_ulysses.apply()`` and
    ``sp_ring.apply()`` from the Phase-1c dispatch loop. Reads the resolved
    ``DiffusionParallelConfig`` and dispatches to the SAME low-level
    installer functions the legacy path uses
    (:func:`apply_sequence_parallel` /
    :func:`apply_sequence_parallel_from_descriptor` in
    ``hooks/sequence_parallel.py``), but does NOT depend on the legacy SP
    runtime helper — the wiring layer between "which transformer(s)" and
    "register hooks" is fully owned by this module.

    Failure-mode contract: this function does NOT swallow exceptions.
    Callers (``sp_ulysses.apply()`` / ``sp_ring.apply()``) wrap the call in
    an identical ``try/except Exception: logger.warning(...)`` sink, so OFF
    and ON paths share warn-and-continue semantics on SP wiring failure
    (N1 still deferred per ``DESIGN_PHASE1C_INIT_DISPATCH.md`` §3.2 N8).

    Args:
        pipeline: the just-constructed pipeline (top-level ``nn.Module``,
            e.g. ``ZImagePipeline``) whose transformer(s) SP hooks should
            be wired onto.
        parallel_config: the resolved
            :class:`vllm_omni.diffusion.data.DiffusionParallelConfig`. Reads
            ``sequence_parallel_size``, ``ulysses_degree``, ``ring_degree``,
            ``use_sp_descriptor``.

    Returns:
        Number of transformer modules SP hooks were applied to. ``0`` if SP
        is inert (``sequence_parallel_size <= 1``), if no candidate
        transformer carries an SP plan/descriptor, or if a previous call
        already wired SP this stage init (re-entry guard via
        :attr:`ForwardContext.sp_plan_hooks_applied`).
    """
    sp_size = parallel_config.sequence_parallel_size
    if sp_size <= 1:
        return 0

    # Re-entry / idempotency guard. In hybrid configs both
    # ``sp_ulysses.apply()`` and ``sp_ring.apply()`` may route here; the
    # first wins. Mirrors the legacy helper's check at
    # ``hooks/sequence_parallel.py:718-720``. NOTE: ``sp_ring.apply()`` ALSO
    # short-circuits at the dispatch boundary (``sp_ring.py:79-85``); this
    # is the belt-and-braces second check.
    from vllm_omni.diffusion.forward_context import get_forward_context

    fc = get_forward_context()
    if fc.sp_plan_hooks_applied:
        return 0

    use_descriptor = getattr(parallel_config, "use_sp_descriptor", False)

    sp_config = SequenceParallelConfig(
        ulysses_degree=parallel_config.ulysses_degree,
        ring_degree=parallel_config.ring_degree,
    )

    # Declarative pipeline-shape lookup. Replaces the legacy hardcoded
    # attribute-name scan + ``find_module_with_attr`` walk at
    # ``hooks/sequence_parallel.py:723-743``.
    targets = get_transformers_for_pipeline(pipeline)
    if not targets:
        # INTENTIONAL DIVERGENCE FROM LEGACY (Phase 1c-Twin TWIN-3 Option A):
        # the new path emits an adapter-specific warning that recommends
        # registering a PipelineTransformerAdapter, which is more
        # actionable than the legacy "no hook-based SP plan/descriptor was
        # applied" text at ``hooks/sequence_parallel.py:795-801``. The
        # equivalence gates do NOT assert log-text equality.
        logger.warning(
            f"Sequence parallelism is enabled (sp_size={sp_size}) but no "
            "transformer was discovered on the pipeline "
            f"({pipeline.__class__.__name__}). This is expected for "
            "manual-SP models that implement sequence parallelism inside "
            "forward() (e.g. an SPInternal-marked model like BAGEL). "
            "Otherwise, consider registering a PipelineTransformerAdapter "
            "for this pipeline class."
        )
        return 0

    applied_count = 0
    for transformer, attr_name in targets:
        if use_descriptor:
            # Typed-descriptor path. SPInternal (Mechanism B) and models
            # without a descriptor return ``False`` from the installer; we
            # mirror the legacy descriptor branch's
            # ``if applied: ...; continue`` shape exactly
            # (``hooks/sequence_parallel.py:753-766``).
            applied = apply_sequence_parallel_from_descriptor(
                transformer, sp_config,
            )
            if applied:
                logger.info(
                    f"Applying sequence parallelism to "
                    f"{transformer.__class__.__name__} ({attr_name}) "
                    f"via SPDescriptor (sp_size={sp_size}, "
                    f"mode={_mode_tag(sp_config)}, "
                    f"ulysses={sp_config.ulysses_degree}, "
                    f"ring={sp_config.ring_degree})"
                )
                applied_count += 1
            continue

        # Legacy ``_sp_plan`` branch (F2 lock-in, per
        # ``DESIGN_PHASE1C_INIT_DISPATCH.md`` §4.5.4): even on
        # descriptor-bearing models, this branch is taken when
        # ``use_sp_descriptor=False``. Lazy import so unit tests can patch
        # ``vllm_omni.diffusion.distributed.sp_plan.get_sp_plan_from_model``
        # at the source module (matches the legacy helper's lazy import
        # pattern at ``hooks/sequence_parallel.py:705``).
        from vllm_omni.diffusion.distributed.sp_plan import (
            get_sp_plan_from_model,
        )

        plan = get_sp_plan_from_model(transformer)
        if plan is None:
            continue
        logger.info(
            f"Applying sequence parallelism to "
            f"{transformer.__class__.__name__} ({attr_name}) "
            f"(sp_size={sp_size}, mode={_mode_tag(sp_config)}, "
            f"ulysses={sp_config.ulysses_degree}, "
            f"ring={sp_config.ring_degree})"
        )
        apply_sequence_parallel(transformer, sp_config, plan)
        applied_count += 1

    fc.sp_plan_hooks_applied = applied_count > 0
    return applied_count
