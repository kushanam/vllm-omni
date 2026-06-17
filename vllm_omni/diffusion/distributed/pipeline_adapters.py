# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declarative pipeline-shape policy for the new (Phase 1c-Twin) SP runtime.

This module replaces the hardcoded list of transformer attribute names
(``["transformer", "transformer_2", "dit", "unet"]``) scanned plus the
``find_module_with_attr`` recursive walk inside the legacy SP runtime helper
in ``vllm_omni.diffusion.hooks.sequence_parallel`` (legacy helper at
``hooks/sequence_parallel.py:723-743``) with a class-based adapter registry.

The shape mirrors the teacache adapter precedent at
``vllm_omni/diffusion/cache/teacache/coefficient_estimator.py`` (see
``DefaultAdapter`` / ``BagelAdapter`` and the module-level ``_MODEL_ADAPTERS``
dict): a base class with a classmethod hook, subclasses for each non-default
pipeline shape, and a module-level ``dict[str, type[...]]`` keyed by
``pipeline.__class__.__name__``. Missing keys fall back to
:class:`DefaultAdapter`.

This is the pipeline-shape policy boundary for the *new* SP runtime path
(``vllm_omni.diffusion.distributed.sp_runtime.apply_sp_to_pipeline``); the
*legacy* ``_apply_sp_runtime`` keeps its hardcoded scan unchanged (Phase
1c-Twin §6 frozen surface).
"""
from __future__ import annotations

import torch.nn as nn


class PipelineTransformerAdapter:
    """Declarative replacement for the legacy hardcoded attribute-name scan.

    Subclasses describe how to enumerate the SP-eligible transformer modules
    of a given pipeline class. Patterned after the teacache
    ``DefaultAdapter`` / ``BagelAdapter`` precedent at
    ``vllm_omni/diffusion/cache/teacache/coefficient_estimator.py``.
    """

    @classmethod
    def get_transformers(cls, pipeline) -> list[tuple[nn.Module, str]]:
        """Return ``[(transformer_module, attribute_name), ...]``.

        The attribute name is log-only — it mirrors the legacy helper's
        ``({attr})`` log fragment at
        ``hooks/sequence_parallel.py:761`` and ``:784``. Return an empty
        list to no-op SP wiring on this pipeline.
        """
        raise NotImplementedError


class DefaultAdapter(PipelineTransformerAdapter):
    """The catch-all adapter.

    Covers the 90%+ of SP-capable pipelines whose only SP-eligible
    transformer is ``pipeline.transformer``. Includes the BAGEL/Lance
    aliasing case where the pipeline sets
    ``self.transformer = self.language_model.model``
    (``vllm_omni/diffusion/models/bagel/pipeline_bagel.py:254``,
    ``pipeline_lance.py:290``); for those models the language-model module
    carries an ``_sp_descriptor = SPInternal()`` declaration and the
    descriptor installer no-ops (Mechanism B).
    """

    @classmethod
    def get_transformers(cls, pipeline) -> list[tuple[nn.Module, str]]:
        t = getattr(pipeline, "transformer", None)
        return [(t, "transformer")] if t is not None else []


class Wan22Adapter(PipelineTransformerAdapter):
    """Two-transformer adapter for the Wan2.2 family.

    ``Wan22Pipeline.__init__`` constructs both transformers (T2V at
    ``pipeline_wan2_2.py:377`` and ``:383``); ``Wan22I2VPipeline.__init__``
    does the same (``pipeline_wan2_2_i2v.py:263`` and ``:278``).
    ``Wan22S2VPipeline`` only sets ``self.transformer``
    (``pipeline_wan2_2_s2v.py:577``); the ``getattr(..., None)`` guard
    transparently handles that case (``transformer_2`` returns ``None`` and
    is skipped). ``Wan22VACEPipeline`` inherits from ``Wan22Pipeline``
    (``pipeline_wan2_2_vace.py:166``) so it gets both.
    """

    @classmethod
    def get_transformers(cls, pipeline) -> list[tuple[nn.Module, str]]:
        out: list[tuple[nn.Module, str]] = []
        t1 = getattr(pipeline, "transformer", None)
        if t1 is not None:
            out.append((t1, "transformer"))
        t2 = getattr(pipeline, "transformer_2", None)
        if t2 is not None:
            out.append((t2, "transformer_2"))
        return out


class LTX2TwoStagesAdapter(PipelineTransformerAdapter):
    """Explicit one-level descent for the LTX2 two-stage pipelines.

    ``LTX2TwoStagesPipeline.__init__`` constructs an inner
    ``self.pipe = LTX2Pipeline(...)`` at
    ``vllm_omni/diffusion/models/ltx2/pipeline_ltx2.py:1173``;
    ``LTX2ImageToVideoTwoStagesPipeline.__init__`` likewise sets
    ``self.pipe = LTX2ImageToVideoPipeline(...)`` at
    ``pipeline_ltx2_image2video.py:763``. The SP-eligible transformer
    lives at ``self.pipe.transformer``. The legacy helper handled this
    via a recursive ``find_module_with_attr`` walk
    (``hooks/sequence_parallel.py:736``); this adapter replaces that walk
    with an explicit one-level descent so the supported shape is
    grep-able and adding a new descent target is an explicit registry
    entry, not a silent ``hasattr`` lookup.
    """

    @classmethod
    def get_transformers(cls, pipeline) -> list[tuple[nn.Module, str]]:
        inner = getattr(pipeline, "pipe", None)
        if inner is None:
            return []
        t = getattr(inner, "transformer", None)
        return [(t, "pipe.transformer")] if t is not None else []


_PIPELINE_TRANSFORMER_ADAPTERS: dict[str, type[PipelineTransformerAdapter]] = {
    # Wan2.2 family: transformer + transformer_2 (transformer_2 may be None
    # on S2V; Wan22Adapter handles that via getattr(..., None)).
    "Wan22Pipeline": Wan22Adapter,
    "Wan22I2VPipeline": Wan22Adapter,
    "Wan22S2VPipeline": Wan22Adapter,
    "Wan22VACEPipeline": Wan22Adapter,
    "WanT2VDMD2Pipeline": Wan22Adapter,
    "WanI2VDMD2Pipeline": Wan22Adapter,
    # LTX2 two-stage pipelines: pipeline.pipe.transformer.
    "LTX2TwoStagesPipeline": LTX2TwoStagesAdapter,
    "LTX2ImageToVideoTwoStagesPipeline": LTX2TwoStagesAdapter,
    # Every other SP-capable pipeline class falls through to DefaultAdapter
    # via get_transformers_for_pipeline() below.
}


def get_transformers_for_pipeline(
    pipeline,
) -> list[tuple[nn.Module, str]]:
    """Public lookup used by :func:`apply_sp_to_pipeline`.

    Looks up an adapter class by ``pipeline.__class__.__name__`` and falls
    back to :class:`DefaultAdapter` when no override is registered. Returns
    an empty list when no SP-eligible transformer is discoverable on the
    pipeline (e.g. the default adapter's ``pipeline.transformer is None``
    case, or the LTX2 two-stage adapter's ``pipeline.pipe is None`` case).
    """
    adapter = _PIPELINE_TRANSFORMER_ADAPTERS.get(
        pipeline.__class__.__name__, DefaultAdapter,
    )
    return adapter.get_transformers(pipeline)
