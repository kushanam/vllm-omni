# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel axis module.

``plan()`` emits the same engine kwarg the translator emits for a ``tp`` axis
(``tensor_parallel_size``) but is execution-type-aware about *ownership*
(``REVIEW_PHASE1_IMPL`` §SHOULD-FIX 1): diffusion ``tp`` is omni-executed
(``owned_by="omni"``, ``rank_token="tp"``) per contract §5, while AR ``tp`` is
delegated to vLLM core (``owned_by="vllm"``). That "diffusion → omni, AR → vLLM"
rule is a *backend* property, so it is no longer branched inline here: the owner
is resolved from the backend's execution-owner table via
:func:`axis_execution_owner` (the single source of truth, vocabulary #2). Only
the ownership view differs — ``engine_kwargs`` is identical in both cases — so
``build_groups()`` / ``apply()`` remain the typed no-ops inherited from
:class:`DelegatedStrategy` in Phase 1b.
"""
from __future__ import annotations

from vllm_omni.config.composable_parallel.backends import (
    VLLM_BACKEND,
    axis_execution_owner,
)
from vllm_omni.config.composable_parallel.modules.base import (
    AxisPlan,
    DelegatedStrategy,
    LoweringCtx,
)


class TensorParallelStrategy(DelegatedStrategy):
    axis = "tp"

    def __init__(self, degree: int):
        self._degree = int(degree)

    def plan(self, ctx: LoweringCtx) -> AxisPlan:
        # Owner comes from the backend's execution-owner table (vocabulary #2),
        # not an inline is_diffusion_execution branch. ``rank_token`` is the
        # DiT-world token only when omni executes it; None when vLLM owns the
        # world (its own RankGenerator) — reproduces the pre-refactor value.
        owner = axis_execution_owner(ctx.backend or VLLM_BACKEND, self.axis, ctx.execution_type)
        return AxisPlan(
            axis="tp",
            degree=self._degree,
            owned_by=owner,
            engine_kwargs={"tensor_parallel_size": self._degree},
            rank_token="tp" if owner == "omni" else None,
        )
