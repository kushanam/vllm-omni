# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VAE patch-parallelism axis module (Phase 1 pilot, §4).

``vae_pp`` is the one Phase-1 module with a real ``apply()``. It does NOT consume
a world dimension and builds no process group: the live VAE-PP path
(``DistributedVaeExecutor`` / ``DistributedVaeMixin``) reuses the **DiT** group
via ``get_dit_group()`` and caps ``vae_patch_parallel_size`` at the DiT world
size (§4.1). ``vae_pp`` is also a reserved kind the translator rejects today
(§8 G1), so there is no translate/apply baseline for it; the equivalence
guarantee lives at the ``apply()`` / registry layer (side-effect identity, §5.3).

``apply()`` routes the inline registry VAE-PP poke through ``_wire_distributed_vae``,
which owns the full vae_pp block (warning + tiling auto-enable + ``set_parallel_size``)
so the tiling-ordering nuance of ``registry.initialize_model`` is preserved (§4.2).
"""
from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.config.composable_parallel.backends import (
    VLLM_BACKEND,
    axis_execution_owner,
)
from vllm_omni.config.composable_parallel.modules.base import (
    ApplyCtx,
    AxisPlan,
    AxisResult,
    GroupBuildCtx,
    GroupHandle,
    LoweringCtx,
    OmniExecutedStrategy,
)

logger = init_logger(__name__)


class VaePatchParallelStrategy(OmniExecutedStrategy):
    axis = "vae_pp"
    # Real init-time apply() (see below): dispatch this module at model init.
    supports_init_dispatch = True
    # Runs FIRST in the init-dispatch loop (was APPLY_ORDER[0]). Rationale: its
    # auto-enable of ``od_config.vae_use_tiling`` must be observed by the
    # registry-side VAE memory-optimization step that runs after dispatch
    # returns (orchestrator §4.4.2). Ordered before the SP axes (10 < 20 < 30).
    init_dispatch_order = 10

    def __init__(self, degree: int):
        self._degree = int(degree)

    def plan(self, ctx: LoweringCtx) -> AxisPlan:
        return AxisPlan(
            axis="vae_pp",
            degree=self._degree,
            owned_by=axis_execution_owner(ctx.backend or VLLM_BACKEND, self.axis, ctx.execution_type),
            engine_kwargs={"vae_patch_parallel_size": self._degree},
            rank_token=None,            # reuses the DiT group
            consumes_world_dim=False,   # capped at DiT world size; adds no ranks
        )

    def build_groups(self, ctx: GroupBuildCtx) -> AxisResult:
        # No new group: VAE-PP reuses the DiT group built by init_dit_group.
        return AxisResult(
            axis="vae_pp",
            group=GroupHandle("vae_pp", "reused", coordinator=None, reused=True),
            notes=("reuses DiT group via get_dit_group()",),
        )

    def apply(self, ctx: ApplyCtx) -> AxisResult:
        # Exactly the poke that registry.initialize_model does today (§4.2).
        return _wire_distributed_vae(ctx.model, ctx.od_config)


def _wire_distributed_vae(model, od_config) -> AxisResult:
    """The exact VAE-PP block from ``registry.initialize_model`` (§4.2).

    Owns the full vae_pp logic so ordering is preserved: the ``vae_use_tiling``
    auto-enable runs BEFORE the caller writes ``model.vae.use_tiling =
    od_config.vae_use_tiling`` (the memory-optimization lines kept inline in the
    registry), so the auto-enabled value is what gets written.
    """
    # Local import to avoid an import-time cycle (registry imports this module).
    from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
        DistributedVaeMixin,
    )

    vae_pp_size = od_config.parallel_config.vae_patch_parallel_size
    is_distributed_vae = hasattr(model, "vae") and isinstance(model.vae, DistributedVaeMixin)
    if vae_pp_size > 1 and not is_distributed_vae:
        logger.warning(
            "vae_patch_parallel_size=%d is set but VAE patch parallelism is NOT enabled for %s; ignoring.",
            vae_pp_size,
            od_config.model_class_name,
        )
    if vae_pp_size > 1 and is_distributed_vae and not od_config.vae_use_tiling:
        logger.info(
            "vae_patch_parallel_size=%d requires vae_use_tiling; automatically enabling it.",
            vae_pp_size,
        )
        od_config.vae_use_tiling = True

    if is_distributed_vae:
        model.vae.set_parallel_size(vae_pp_size)

    return AxisResult(axis="vae_pp")
