# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models.registry import _LazyRegisteredModel, _ModelRegistry

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.hooks.sequence_parallel import _apply_sp_runtime
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_DIFFUSION_MODELS = {
    # arch:(mod_folder, mod_relname, cls_name)
    "QwenImagePipeline": (
        "qwen_image",
        "pipeline_qwen_image",
        "QwenImagePipeline",
    ),
    "QwenImageEditPipeline": (
        "qwen_image",
        "pipeline_qwen_image_edit",
        "QwenImageEditPipeline",
    ),
    "QwenImageEditPlusPipeline": (
        "qwen_image",
        "pipeline_qwen_image_edit_plus",
        "QwenImageEditPlusPipeline",
    ),
    "QwenImageLayeredPipeline": (
        "qwen_image",
        "pipeline_qwen_image_layered",
        "QwenImageLayeredPipeline",
    ),
    "GlmImagePipeline": (
        "glm_image",
        "pipeline_glm_image",
        "GlmImagePipeline",
    ),
    "ZImagePipeline": (
        "z_image",
        "pipeline_z_image",
        "ZImagePipeline",
    ),
    "OvisImagePipeline": (
        "ovis_image",
        "pipeline_ovis_image",
        "OvisImagePipeline",
    ),
    "WanPipeline": (
        "wan2_2",
        "pipeline_wan2_2",
        "Wan22Pipeline",
    ),
    "WanVACEPipeline": (
        "wan2_2",
        "pipeline_wan2_2_vace",
        "Wan22VACEPipeline",
    ),
    "LTX2Pipeline": (
        "ltx2",
        "pipeline_ltx2",
        "LTX2Pipeline",
    ),
    "LTX2ImageToVideoPipeline": (
        "ltx2",
        "pipeline_ltx2_image2video",
        "LTX2ImageToVideoPipeline",
    ),
    "LTX2TwoStagesPipeline": (
        "ltx2",
        "pipeline_ltx2",
        "LTX2TwoStagesPipeline",
    ),
    "LTX2ImageToVideoTwoStagesPipeline": (
        "ltx2",
        "pipeline_ltx2_image2video",
        "LTX2ImageToVideoTwoStagesPipeline",
    ),
    "LTX2T2VDMD2Pipeline": (
        "ltx2",
        "pipeline_ltx2",
        "LTX2T2VDMD2Pipeline",
    ),
    "LTX2I2VDMD2Pipeline": (
        "ltx2",
        "pipeline_ltx2_image2video",
        "LTX2I2VDMD2Pipeline",
    ),
    "LTX23Pipeline": (
        "ltx2",
        "pipeline_ltx2_3",
        "LTX23Pipeline",
    ),
    "LTX23ImageToVideoPipeline": (
        "ltx2",
        "pipeline_ltx2_3_image2video",
        "LTX23ImageToVideoPipeline",
    ),
    "StableAudioPipeline": (
        "stable_audio",
        "pipeline_stable_audio",
        "StableAudioPipeline",
    ),
    "WanImageToVideoPipeline": (
        "wan2_2",
        "pipeline_wan2_2_i2v",
        "Wan22I2VPipeline",
    ),
    "WanS2VPipeline": (
        "wan2_2",
        "pipeline_wan2_2_s2v",
        "Wan22S2VPipeline",
    ),
    "WanT2VDMD2Pipeline": (
        "wan2_2",
        "pipeline_wan2_2",
        "WanT2VDMD2Pipeline",
    ),
    "WanI2VDMD2Pipeline": (
        "wan2_2",
        "pipeline_wan2_2_i2v",
        "WanI2VDMD2Pipeline",
    ),
    "LongCatImagePipeline": (
        "longcat_image",
        "pipeline_longcat_image",
        "LongCatImagePipeline",
    ),
    "BagelPipeline": (
        "bagel",
        "pipeline_bagel",
        "BagelPipeline",
    ),
    "LancePipeline": (
        "lance",
        "pipeline_lance",
        "LancePipeline",
    ),
    "MingImagePipeline": (
        "ming_flash_omni",
        "pipeline_ming_imagegen",
        "MingImagePipeline",
    ),
    "InternVLAA1Pipeline": (
        "internvla_a1",
        "pipeline_internvla_a1",
        "InternVLAA1Pipeline",
    ),
    "Gr00tN1d7Pipeline": (
        "gr00t",
        "pipeline_gr00t",
        "Gr00tN1d7Pipeline",
    ),
    "LongCatImageEditPipeline": (
        "longcat_image",
        "pipeline_longcat_image_edit",
        "LongCatImageEditPipeline",
    ),
    "StableDiffusion3Pipeline": (
        "sd3",
        "pipeline_sd3",
        "StableDiffusion3Pipeline",
    ),
    "FluxKontextPipeline": (
        "flux",
        "pipeline_flux_kontext",
        "FluxKontextPipeline",
    ),
    "HunyuanImage3ForCausalMM": (
        "hunyuan_image3",
        "pipeline_hunyuan_image3",
        "HunyuanImage3Pipeline",
    ),
    "Flux2KleinPipeline": (
        "flux2_klein",
        "pipeline_flux2_klein",
        "Flux2KleinPipeline",
    ),
    "ErnieImagePipeline": (
        "ernie_image",
        "pipeline_ernie_image",
        "ErnieImagePipeline",
    ),
    "NextStep11Pipeline": (
        "nextstep_1_1",
        "pipeline_nextstep_1_1",
        "NextStep11Pipeline",
    ),
    "FluxPipeline": (
        "flux",
        "pipeline_flux",
        "FluxPipeline",
    ),
    "FluxDMD2Pipeline": (
        "flux",
        "pipeline_flux",
        "FluxDMD2Pipeline",
    ),
    "QwenImageDMD2Pipeline": (
        "qwen_image",
        "pipeline_qwen_image",
        "QwenImageDMD2Pipeline",
    ),
    "OmniGen2Pipeline": (
        "omnigen2",
        "pipeline_omnigen2",
        "OmniGen2Pipeline",
    ),
    "HeliosPipeline": (
        "helios",
        "pipeline_helios",
        "HeliosPipeline",
    ),
    "HeliosPyramidPipeline": (
        "helios",
        "pipeline_helios",
        "HeliosPipeline",
    ),
    "Flux2Pipeline": (
        "flux2",
        "pipeline_flux2",
        "Flux2Pipeline",
    ),
    "DreamIDOmniPipeline": (
        "dreamid_omni",
        "pipeline_dreamid_omni",
        "DreamIDOmniPipeline",
    ),
    "SenseNovaU1Pipeline": (
        "sensenova_u1",
        "pipeline_sensenova_u1",
        "SenseNovaU1Pipeline",
    ),
    "AudioXPipeline": (
        "audiox",
        "pipeline_audiox",
        "AudioXPipeline",
    ),
    "HunyuanVideo15Pipeline": (
        "hunyuan_video",
        "pipeline_hunyuan_video_1_5",
        "HunyuanVideo15Pipeline",
    ),
    "HunyuanVideo15ImageToVideoPipeline": (
        "hunyuan_video",
        "pipeline_hunyuan_video_1_5_i2v",
        "HunyuanVideo15I2VPipeline",
    ),
    "MagiHumanPipeline": (
        "magi_human",
        "pipeline_magi_human",
        "MagiHumanPipeline",
    ),
    "OmniVoicePipeline": (
        "omnivoice",
        "pipeline_omnivoice",
        "OmniVoicePipeline",
    ),
    "OmniVoice": (
        "omnivoice",
        "pipeline_omnivoice",
        "OmniVoicePipeline",
    ),
    "Cosmos3OmniDiffusersPipeline": (
        "cosmos3",
        "pipeline_cosmos3",
        "Cosmos3OmniDiffusersPipeline",
    ),
    "SoulXSingerPipeline": (
        "soulx_singer",
        "pipeline_soulx_singer_svs",
        "PipelineSoulXSingerSVS",
    ),
    "SoulXSingerSVCPipeline": (
        "soulx_singer",
        "pipeline_soulx_singer_svc",
        "PipelineSoulXSingerSVC",
    ),
    "DiffusersAdapterPipeline": (
        "diffusers_adapter",
        "pipeline_diffusers_adapter",
        "DiffusersAdapterPipeline",
    ),
    "HiDreamImagePipeline": (
        "hidream_image",
        "pipeline_hidream_image",
        "HiDreamImagePipeline",
    ),
    "DreamZeroPipeline": (
        "dreamzero",
        "pipeline_dreamzero",
        "DreamZeroPipeline",
    ),
    "StableDiffusionXLPipeline": (
        "sdxl",
        "pipeline_sdxl",
        "StableDiffusionXLPipeline",
    ),
}


DiffusionModelRegistry = _ModelRegistry(
    {
        model_arch: _LazyRegisteredModel(
            module_name=f"vllm_omni.diffusion.models.{mod_folder}.{mod_relname}",
            class_name=cls_name,
        )
        for model_arch, (mod_folder, mod_relname, cls_name) in _DIFFUSION_MODELS.items()
    }
)

_NO_CACHE_ACCELERATION = {
    # Pipelines that do not support cache acceleration (cache_dit / tea_cache).
    "NextStep11Pipeline",
    "AudioXPipeline",
}


def _prepare_diffusion_quant_config(
    od_config: OmniDiffusionConfig,
    model_class: type[nn.Module],
) -> None:
    """Prepare diffusion quant config using vLLM-style model bindings."""
    quant_config = getattr(od_config, "quantization_config", None)
    if quant_config is None:
        return
    if hasattr(quant_config, "maybe_update_config"):
        quant_config.maybe_update_config(od_config.model)
    diffusion_packed_modules_mapping = current_omni_platform.get_diffusion_packed_modules_mapping(model_class)
    if diffusion_packed_modules_mapping is not None:
        model_class.packed_modules_mapping = diffusion_packed_modules_mapping
    configure_quant_config(quant_config, model_class)


def initialize_model(
    od_config: OmniDiffusionConfig,
) -> nn.Module:
    """Initialize a diffusion model from the registry.

    This function:
    1. Loads the model class from the registry
    2. Instantiates the model with the config
    3. Configures VAE optimization settings
    4. Applies sequence parallelism if enabled (similar to diffusers' enable_parallelism)

    Args:
        od_config: The OmniDiffusion configuration.

    Returns:
        The initialized pipeline model.

    Raises:
        ValueError: If the model class is not found in the registry.
    """
    model_class = DiffusionModelRegistry._try_load_model_cls(od_config.model_class_name)
    if model_class is not None:
        _prepare_diffusion_quant_config(od_config, model_class)
        with set_current_diffusion_config(od_config):
            model = model_class(od_config=od_config)

        # Phase 1c §5.4 R1: VAE-PP and SP wiring are flag-gated. ``ApplyCtx``
        # is imported once here because both branches construct one. The
        # ordering invariants the spec mandates:
        #   * vae_pp.apply() runs BEFORE the ``model.vae.use_tiling = ...``
        #     write so the auto-enabled ``od_config.vae_use_tiling`` value is
        #     propagated onto the model (§4.2 / §4.4.2).
        #   * On the OFF path the today-verbatim order is preserved
        #     (vae_pp -> memory-opt -> legacy SP), keeping the legacy path
        #     byte-identical to pre-1c (G5 / §5.4 R1(a)).
        #   * On the ON path the dispatch loop owns vae_pp + SP (in
        #     APPLY_ORDER); memory-opt runs immediately AFTER the dispatch
        #     returns (post-dispatch normalization, §4.4.2 / §5.4 R1(b)).
        # Imported locally to avoid an import-time cycle.
        from vllm_omni.config.composable_parallel.modules.base import ApplyCtx

        if od_config.parallel_config.use_init_dispatch:
            # New path (§4.2 / §5.4 R1(a)): Orchestrator.apply dispatches the
            # init-dispatchable axes (vae_pp first, then sp_ring/sp_ulysses
            # per APPLY_ORDER). VAE-PP and SP are reached ONLY through this
            # call when the flag is True; the bespoke inline calls below
            # (else-branch) are intentionally NOT executed here so VAE-PP is
            # applied exactly once.
            from vllm_omni.config.composable_parallel.modules.orchestrator import (
                Orchestrator,
            )
            from vllm_omni.config.stage_config import StageExecutionType

            # §10 Assumption 1: ``initialize_model`` is, by construction, only
            # called for diffusion-side model loading. The AR path uses vLLM's
            # own model registry; the broader caller census (loader / ROCm
            # patch / examples) all flow through the diffusion init path.
            exec_type = StageExecutionType.DIFFUSION
            ctx = ApplyCtx(
                model=model,
                od_config=od_config,
                execution_type=exec_type,
            )
            plan = Orchestrator().lower_from_runtime_kwargs(od_config, exec_type)
            Orchestrator().apply(plan, ctx)

            # Memory-opt: post-dispatch normalization (§4.4.2). Runs AFTER
            # vae_pp.apply() returns so the auto-enabled
            # ``od_config.vae_use_tiling`` value is propagated onto
            # ``model.vae.use_tiling``. SP only mutates transformer hooks and
            # does not interact with these VAE knobs.
            if hasattr(model, "vae") and hasattr(model.vae, "use_slicing"):
                model.vae.use_slicing = od_config.vae_use_slicing
            if hasattr(model, "vae") and hasattr(model.vae, "use_tiling"):
                model.vae.use_tiling = od_config.vae_use_tiling
        else:
            # Legacy inline path (§5.4 R1(a) "else run the existing inline
            # paths verbatim"): vae_pp -> memory-opt -> legacy SP wrapper.
            # The thin ``_apply_sequence_parallel_if_enabled`` wrapper
            # preserves Phase-1b warn-and-continue failure-mode parity (N1
            # deferred — see §3.2 N8 / §4.5.4 row 6).
            from vllm_omni.config.composable_parallel.modules.axes.vae_pp import (
                VaePatchParallelStrategy,
            )

            VaePatchParallelStrategy(
                od_config.parallel_config.vae_patch_parallel_size
            ).apply(ApplyCtx(model=model, od_config=od_config))

            if hasattr(model, "vae") and hasattr(model.vae, "use_slicing"):
                model.vae.use_slicing = od_config.vae_use_slicing
            if hasattr(model, "vae") and hasattr(model.vae, "use_tiling"):
                model.vae.use_tiling = od_config.vae_use_tiling

            _apply_sequence_parallel_if_enabled(model, od_config)

        return model
    else:
        raise ValueError(f"Model class {od_config.model_class_name} not found in diffusion model registry.")


def _apply_sequence_parallel_if_enabled(model, od_config: OmniDiffusionConfig) -> int:
    """Apply sequence parallelism hooks if SP is enabled (legacy OFF path).

    Phase 1c §5.4 R1(c): this is now a thin wrapper around the shared
    :func:`_apply_sp_runtime` helper extracted to
    ``vllm_omni.diffusion.hooks.sequence_parallel``. The wrapper preserves the
    Phase-1b ``try/except Exception`` sink (N1 deferred — see
    ``DESIGN_PHASE1C_INIT_DISPATCH.md`` §3.2 N8 / §4.5.4 row 6) so OFF and ON
    paths share bit-identical warn-and-continue failure-mode semantics on SP
    wiring failure. The helper itself never swallows; the failure-mode parity
    is provided HERE (OFF path) and inside each SP module's ``apply()`` body
    (ON path, §5.5 A4).

    Note: Our "Sequence Parallelism" (SP) corresponds to "Context Parallelism"
    (CP) in diffusers. We use ``_sp_plan`` / ``_sp_descriptor`` instead of
    diffusers' ``_cp_plan``.

    Args:
        model: The pipeline model (e.g., ``ZImagePipeline``).
        od_config: The OmniDiffusion configuration.

    Returns:
        Number of transformer modules SP hooks were applied to (0 on
        warn-and-continue failure or when SP is inert).
    """
    try:
        applied_count = _apply_sp_runtime(model, od_config.parallel_config)
    except Exception as e:
        logger.warning(
            f"Failed to apply sequence parallelism: {e}. "
            "Continuing without SP hooks."
        )
        applied_count = 0
    return applied_count


_DIFFUSION_POST_PROCESS_FUNCS = {
    # arch: post_process_func
    # `post_process_func` function must be placed in {mod_folder}/{mod_relname}.py,
    # where mod_folder and mod_relname are  defined and mapped using `_DIFFUSION_MODELS` via the `arch` key
    "QwenImagePipeline": "get_qwen_image_post_process_func",
    "QwenImageEditPipeline": "get_qwen_image_edit_post_process_func",
    "QwenImageEditPlusPipeline": "get_qwen_image_edit_plus_post_process_func",
    "GlmImagePipeline": "get_glm_image_post_process_func",
    "ZImagePipeline": "get_post_process_func",
    "OvisImagePipeline": "get_ovis_image_post_process_func",
    "WanPipeline": "get_wan22_post_process_func",
    "WanVACEPipeline": "get_wan22_vace_post_process_func",
    "LTX2Pipeline": "get_ltx2_post_process_func",
    "LTX2TwoStagesPipeline": "get_ltx2_post_process_func",
    "LTX2ImageToVideoPipeline": "get_ltx2_post_process_func",
    "LTX2ImageToVideoTwoStagesPipeline": "get_ltx2_post_process_func",
    "LTX2T2VDMD2Pipeline": "get_ltx2_post_process_func",
    "LTX2I2VDMD2Pipeline": "get_ltx2_post_process_func",
    "LTX23Pipeline": "get_ltx2_post_process_func",
    "LTX23ImageToVideoPipeline": "get_ltx2_post_process_func",
    "StableAudioPipeline": "get_stable_audio_post_process_func",
    "SoulXSingerPipeline": "get_soulxsinger_post_process_func",
    "SoulXSingerSVCPipeline": "get_soulxsinger_post_process_func",
    "AudioXPipeline": "get_audiox_post_process_func",
    "WanImageToVideoPipeline": "get_wan22_i2v_post_process_func",
    "WanS2VPipeline": "get_wan22_s2v_post_process_func",
    "WanT2VDMD2Pipeline": "get_wan22_post_process_func",
    "WanI2VDMD2Pipeline": "get_wan22_i2v_post_process_func",
    "LongCatImagePipeline": "get_longcat_image_post_process_func",
    "BagelPipeline": "get_bagel_post_process_func",
    "LancePipeline": "get_lance_post_process_func",
    "MingImagePipeline": "get_ming_image_post_process_func",
    "InternVLAA1Pipeline": "get_internvla_a1_post_process_func",
    "LongCatImageEditPipeline": "get_longcat_image_post_process_func",
    "StableDiffusion3Pipeline": "get_sd3_image_post_process_func",
    "FluxKontextPipeline": "get_flux_kontext_post_process_func",
    "Flux2KleinPipeline": "get_flux2_klein_post_process_func",
    "ErnieImagePipeline": "get_ernie_image_post_process_func",
    "NextStep11Pipeline": "get_nextstep11_post_process_func",
    "FluxPipeline": "get_flux_post_process_func",
    "FluxDMD2Pipeline": "get_flux_post_process_func",
    "QwenImageDMD2Pipeline": "get_qwen_image_post_process_func",
    "OmniGen2Pipeline": "get_omnigen2_post_process_func",
    "HeliosPipeline": "get_helios_post_process_func",
    "HeliosPyramidPipeline": "get_helios_post_process_func",
    "Flux2Pipeline": "get_flux2_post_process_func",
    "HunyuanVideo15Pipeline": "get_hunyuan_video_15_post_process_func",
    "HunyuanVideo15ImageToVideoPipeline": "get_hunyuan_video_15_i2v_post_process_func",
    "MagiHumanPipeline": "get_magi_human_post_process_func",
    "OmniVoicePipeline": "get_omnivoice_post_process_func",
    "DreamIDOmniPipeline": "get_dreamid_omni_post_process_func",
    "SenseNovaU1Pipeline": "get_sensenova_u1_post_process_func",
    "Cosmos3OmniDiffusersPipeline": "get_cosmos3_post_process_func",
    "HiDreamImagePipeline": "get_hidream_image_post_process_func",
    "StableDiffusionXLPipeline": "get_sdxl_image_post_process_func",
}

_DIFFUSION_ACTION_POST_PROCESS_FUNCS = {
    # arch: action_post_process_func
    # `action_post_process_func` function must be placed in {mod_folder}/{mod_relname}.py,
    # where mod_folder and mod_relname are defined and mapped using `_DIFFUSION_MODELS` via the `arch` key.
    "Cosmos3OmniDiffusersPipeline": "get_cosmos3_action_post_process_func",
}

_DIFFUSION_IR_OP_PRIORITY_FUNCS = {
    # arch: ir_op_priority_func
    # `ir_op_priority_func` function must be placed in {mod_folder}/{mod_relname}.py,
    # where mod_folder and mod_relname are defined and mapped using `_DIFFUSION_MODELS` via the `arch` key.
    "Cosmos3OmniDiffusersPipeline": "get_cosmos3_ir_op_priority_func",
}

_DIFFUSION_PRE_PROCESS_FUNCS = {
    # arch: pre_process_func
    # `pre_process_func` function must be placed in {mod_folder}/{mod_relname}.py,
    # where mod_folder and mod_relname are  defined and mapped using `_DIFFUSION_MODELS` via the `arch` key
    "GlmImagePipeline": "get_glm_image_pre_process_func",
    "QwenImageEditPipeline": "get_qwen_image_edit_pre_process_func",
    "QwenImageEditPlusPipeline": "get_qwen_image_edit_plus_pre_process_func",
    "LongCatImageEditPipeline": "get_longcat_image_edit_pre_process_func",
    "QwenImageLayeredPipeline": "get_qwen_image_layered_pre_process_func",
    "WanPipeline": "get_wan22_pre_process_func",
    "WanVACEPipeline": "get_wan22_vace_pre_process_func",
    "WanImageToVideoPipeline": "get_wan22_i2v_pre_process_func",
    "WanS2VPipeline": "get_wan22_s2v_pre_process_func",
    "WanT2VDMD2Pipeline": "get_wan22_pre_process_func",
    "WanI2VDMD2Pipeline": "get_wan22_i2v_pre_process_func",
    "OmniGen2Pipeline": "get_omnigen2_pre_process_func",
    "HeliosPipeline": "get_helios_pre_process_func",
    "HeliosPyramidPipeline": "get_helios_pre_process_func",
    "HunyuanVideo15ImageToVideoPipeline": "get_hunyuan_video_15_i2v_pre_process_func",
    "HunyuanImage3ForCausalMM": "get_hunyuan_image_3_pre_process_func",
    "MagiHumanPipeline": "get_magi_human_pre_process_func",
    "Cosmos3OmniDiffusersPipeline": "get_cosmos3_pre_process_func",
    "SoulXSingerPipeline": "get_soulxsinger_pre_process_func",
    "SoulXSingerSVCPipeline": "get_soulxsinger_svc_pre_process_func",
}


def register_diffusion_model(
    model_arch: str,
    module_name: str,
    class_name: str,
    pre_process_func_name: str | None = None,
    post_process_func_name: str | None = None,
    action_post_process_func_name: str | None = None,
    ir_op_priority_func_name: str | None = None,
) -> None:
    """Register a diffusion model pipeline from an out-of-tree plugin.

    This can be used to add new model architectures or to replace an
    existing built-in pipeline with a platform-optimised implementation
    (same ``model_arch`` key).

    Args:
        model_arch: Architecture name (e.g. ``"WanPipeline"``).
        module_name: Fully qualified module path
            (e.g. ``"my_plugin.diffusion.pipeline_wan"``).
        class_name: Class name within *module_name*.
        pre_process_func_name: Optional name of the pre-process function
            located in *module_name*.  Pass ``None`` to keep the existing
            entry when replacing a built-in model.
        post_process_func_name: Optional name of the post-process function
            located in *module_name*.  Pass ``None`` to keep the existing
            entry when replacing a built-in model.
        action_post_process_func_name: Optional name of the action post-process
            function located in *module_name*.  Pass ``None`` to keep the
            existing entry when replacing a built-in model.
        ir_op_priority_func_name: Optional name of the IR op priority merge
            function located in *module_name*. Pass ``None`` to keep the
            existing entry when replacing a built-in model.
    """
    # Register model class in DiffusionModelRegistry
    DiffusionModelRegistry.register_model(
        model_arch,
        f"{module_name}:{class_name}",
    )

    # Store in _DIFFUSION_MODELS so _load_process_func can resolve the
    # module.  Convention: when mod_relname is empty the mod_folder field
    # stores a *full* module path instead of a relative folder.
    _DIFFUSION_MODELS[model_arch] = (module_name, "", class_name)

    # Optionally register pre/post process funcs.
    if pre_process_func_name is not None:
        _DIFFUSION_PRE_PROCESS_FUNCS[model_arch] = pre_process_func_name
    if post_process_func_name is not None:
        _DIFFUSION_POST_PROCESS_FUNCS[model_arch] = post_process_func_name
    if action_post_process_func_name is not None:
        _DIFFUSION_ACTION_POST_PROCESS_FUNCS[model_arch] = action_post_process_func_name
    if ir_op_priority_func_name is not None:
        _DIFFUSION_IR_OP_PRIORITY_FUNCS[model_arch] = ir_op_priority_func_name

    logger.info(
        "Registered diffusion model %s -> %s.%s",
        model_arch,
        module_name,
        class_name,
    )


def _load_process_func(od_config: OmniDiffusionConfig, func_name: str):
    """Load and return a process function from the appropriate module."""
    mod_folder, mod_relname, _ = _DIFFUSION_MODELS[od_config.model_class_name]
    if mod_relname == "":
        # Full module path (registered via register_diffusion_model)
        module_name = mod_folder
    else:
        # Built-in model (relative path convention)
        module_name = f"vllm_omni.diffusion.models.{mod_folder}.{mod_relname}"
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return func(od_config)


def get_diffusion_post_process_func(od_config: OmniDiffusionConfig):
    if od_config.model_class_name not in _DIFFUSION_POST_PROCESS_FUNCS:
        return None
    func_name = _DIFFUSION_POST_PROCESS_FUNCS[od_config.model_class_name]
    return _load_process_func(od_config, func_name)


def get_diffusion_action_post_process_func(od_config: OmniDiffusionConfig):
    if od_config.model_class_name not in _DIFFUSION_ACTION_POST_PROCESS_FUNCS:
        return None
    func_name = _DIFFUSION_ACTION_POST_PROCESS_FUNCS[od_config.model_class_name]
    return _load_process_func(od_config, func_name)


def get_diffusion_ir_op_priority_func(od_config: OmniDiffusionConfig):
    if od_config.model_class_name not in _DIFFUSION_IR_OP_PRIORITY_FUNCS:
        return None
    func_name = _DIFFUSION_IR_OP_PRIORITY_FUNCS[od_config.model_class_name]
    return _load_process_func(od_config, func_name)


def get_diffusion_pre_process_func(od_config: OmniDiffusionConfig):
    if od_config.model_class_name not in _DIFFUSION_PRE_PROCESS_FUNCS:
        return None  # Return None if no pre-processing function is registered (for backward compatibility)
    func_name = _DIFFUSION_PRE_PROCESS_FUNCS[od_config.model_class_name]
    return _load_process_func(od_config, func_name)
