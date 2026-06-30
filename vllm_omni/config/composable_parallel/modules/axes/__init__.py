# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-axis StrategyModule implementations.

One file per mesh-axis kind. Phase 1 ships the translatable axes wired by the
T2 equivalence matrix (tp/dp/pp/ep/sp_ulysses/sp_ring/stage_replica). VAE-PP
lands in T3; other reserved kinds arrive in later phases.

``STRATEGY_MODULE_BY_KIND`` maps a ``MeshAxisKind`` (== the module ``AxisName``)
to its module class, used by the orchestrator to build the module view from a
``StrategySpec`` stack.
"""
from __future__ import annotations

from vllm_omni.config.composable_parallel.modules.axes.dp import DataParallelStrategy
from vllm_omni.config.composable_parallel.modules.axes.ep import ExpertParallelStrategy
from vllm_omni.config.composable_parallel.modules.axes.pp import (
    PipelineParallelStrategy,
)
from vllm_omni.config.composable_parallel.modules.axes.sp_ring import (
    RingSequenceParallelStrategy,
)
from vllm_omni.config.composable_parallel.modules.axes.sp_ulysses import (
    UlyssesSequenceParallelStrategy,
)
from vllm_omni.config.composable_parallel.modules.axes.stage_replica import (
    StageReplicaStrategy,
)
from vllm_omni.config.composable_parallel.modules.axes.tp import TensorParallelStrategy
from vllm_omni.config.composable_parallel.modules.axes.vae_pp import (
    VaePatchParallelStrategy,
)

# Canonical per-axis StrategyModule classes (one per mesh-axis kind). The
# single source of truth for capability introspection: the backend
# ``effective_init_dispatch_axes`` helper and the §5.1 anti-drift consistency
# test both enumerate THIS tuple rather than a hand-maintained allowlist.
STRATEGY_MODULE_CLASSES = (
    TensorParallelStrategy,
    DataParallelStrategy,
    PipelineParallelStrategy,
    ExpertParallelStrategy,
    UlyssesSequenceParallelStrategy,
    RingSequenceParallelStrategy,
    StageReplicaStrategy,
    VaePatchParallelStrategy,
)

__all__ = [
    "TensorParallelStrategy",
    "DataParallelStrategy",
    "PipelineParallelStrategy",
    "ExpertParallelStrategy",
    "UlyssesSequenceParallelStrategy",
    "RingSequenceParallelStrategy",
    "StageReplicaStrategy",
    "VaePatchParallelStrategy",
    "STRATEGY_MODULE_CLASSES",
]
