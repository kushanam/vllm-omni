# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The canonical internal form: StrategyModule contract types.

Phase 1 ships the contract types (``base``); the orchestrator and per-axis
modules land in T2/T3.
"""

from vllm_omni.config.composable_parallel.modules.base import (
    ApplyCtx,
    AxisName,
    AxisPlan,
    AxisResult,
    DelegatedStrategy,
    GroupBuildCtx,
    GroupHandle,
    LoweringCtx,
    OmniExecutedStrategy,
    OwnedBy,
    StrategyModule,
)

__all__ = [
    "AxisName",
    "OwnedBy",
    "AxisPlan",
    "GroupHandle",
    "AxisResult",
    "LoweringCtx",
    "GroupBuildCtx",
    "ApplyCtx",
    "StrategyModule",
    "OmniExecutedStrategy",
    "DelegatedStrategy",
]
