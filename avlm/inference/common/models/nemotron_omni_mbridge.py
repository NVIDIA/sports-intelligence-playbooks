"""Megatron-Bridge inference adapter for Nemotron-3 Nano Omni."""

import os
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoProcessor

from megatron.bridge import AutoBridge
from megatron.bridge.models.nemotron_omni.nemotron_omni_utils import compute_mel_features, load_audio
from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import adjust_image_tokens
from megatron.bridge.peft.utils import create_peft, create_peft_hook, load_peft_adapter_checkpoint
from megatron.bridge.training.utils.checkpoint_utils import read_run_config

from avlm.inference.common.models import InferenceDistribution, InferenceModel
from avlm.inference.common.utils.video_io import maybe_path_or_url_to_data_urls, pil_image_from_base64


_PATCH_DIM = 16
_AUDIO_SAMPLE_RATE = 16_000
_NUM_MEL_BINS=128


def _apply_fsdp_overrides():
    from megatron.bridge.recipes.nemotron_omni.bridge_integration.megatron_fsdp_buffer_index import (
        apply_megatron_fsdp_buffer_index_patch,
    )
    from megatron.bridge.recipes.nemotron_omni.bridge_integration.megatron_fsdp_mamba_checkpoint import (
        apply_megatron_fsdp_mamba_checkpoint_patch,
    )
    from megatron.bridge.recipes.nemotron_omni.bridge_integration.nemotron_omni_model_config import (
        apply_nemotron_omni_model_config_override,
    )
    from megatron.bridge.recipes.nemotron_omni.bridge_integration.peft_fsdp_pretrained_load import (
        apply_peft_fsdp_pretrained_load_patch,
    )

    apply_nemotron_omni_model_config_override()
    apply_megatron_fsdp_buffer_index_patch()
    apply_megatron_fsdp_mamba_checkpoint_patch()
    apply_peft_fsdp_pretrained_load_patch()


def _freeze_models(models):
    for model in models:
        model.requires_grad_(False)
    return models


def _load_fsdp_model(provider, model_path, sharding_strategy, peft=None, adapter_path=None):
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.bridge.training.checkpointing import load_checkpoint
    from megatron.bridge.training.config import (
        CheckpointConfig,
        ConfigContainer,
        LoggerConfig,
        OptimizerConfig,
    )
    from megatron.bridge.training.state import GlobalState

    provider.params_dtype = torch.bfloat16
    provider.gradient_accumulation_fusion = False
    provider.finalize()

    if sharding_strategy not in {"optim_grads_params", "no_shard"}:
        raise ValueError(
            "fsdp_sharding_strategy must be 'optim_grads_params' or 'no_shard', "
            f"got {sharding_strategy!r}"
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    num_distributed_optimizer_instances = (
        world_size // local_world_size if sharding_strategy == "optim_grads_params" else 1
    )
    provider.initialize_model_parallel(
        seed=0,
        num_distributed_optimizer_instances=num_distributed_optimizer_instances,
    )

    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        check_for_nan_in_grad=True,
        use_megatron_fsdp=True,
        data_parallel_sharding_strategy=sharding_strategy,
        megatron_fsdp_main_params_dtype=None,
        num_distributed_optimizer_instances=num_distributed_optimizer_instances,
    )
    state = GlobalState()
    state.cfg = ConfigContainer(
        model=provider,
        train=None,
        optimizer=OptimizerConfig(use_distributed_optimizer=False),
        ddp=ddp_config,
        scheduler=None,
        dataset=None,
        logger=LoggerConfig(),
        tokenizer=None,
        checkpoint=CheckpointConfig(
            load=model_path,
            finetune=True,
            load_optim=False,
            load_rng=False,
            ckpt_format="fsdp_dtensor",
        ),
        peft=peft,
        dist=None,
    )

    def load_base_checkpoint(models):
        load_checkpoint(
            state=state,
            model=models,
            optimizer=None,
            opt_param_scheduler=None,
            strict=True,
        )
        return models

    pre_wrap_hooks = []
    if peft is not None:
        pre_wrap_hooks.extend([load_base_checkpoint, create_peft_hook(peft, training=False)])
    elif sharding_strategy == "no_shard":
        pre_wrap_hooks.append(_freeze_models)

    model = provider.provide_distributed_model(
        ddp_config=ddp_config,
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
        overlap_param_gather_with_optimizer_step=False,
        data_parallel_random_init=False,
        pre_wrap_hook=pre_wrap_hooks or None,
    )

    if peft is None:
        load_checkpoint(
            state=state,
            model=model,
            optimizer=None,
            opt_param_scheduler=None,
            strict=True,
        )
    else:
        state.cfg.checkpoint.load = adapter_path
        load_checkpoint(
            state=state,
            model=model,
            optimizer=None,
            opt_param_scheduler=None,
            strict=False,
        )
    return model


def _patchify_video(frames, patch_dim):
    """Convert HF-normalized frames to the packed RADIO patch layout."""
    num_frames, channels, height, width = frames.shape
    patches_y = height // patch_dim
    patches_x = width // patch_dim
    patches = (
        frames.contiguous()
        .reshape(num_frames, channels, patches_y, patch_dim, patches_x, patch_dim)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(1, num_frames * patches_y * patches_x, channels * patch_dim * patch_dim)
        .contiguous()
    )
    sizes = torch.tensor([[height, width]] * num_frames, dtype=torch.long)
    return patches, sizes


def _build_vision_packed_seq_params(imgs_sizes):
    lengths = [
        (int(height) // _PATCH_DIM) * (int(width) // _PATCH_DIM)
        for height, width in imgs_sizes.tolist()
    ]
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    cu_seqlens = torch.tensor(cumulative, dtype=torch.int32, device=imgs_sizes.device)
    max_seqlen = torch.tensor(max(lengths), dtype=torch.int32, device=imgs_sizes.device)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


def _align_sound_tokens(input_ids, sound_token_id, desired_count):
    mask = input_ids[0] == sound_token_id
    positions = torch.where(mask)[0]
    if positions.numel() == 0:
        raise ValueError(f"No sound tokens (id={sound_token_id}) found in input_ids")
    current_count = int(positions.numel())
    if current_count == desired_count:
        return input_ids
    start = int(positions[0].item())
    end = int(positions[-1].item()) + 1
    if (end - start) != current_count:
        raise ValueError("Sound tokens are not contiguous; cannot safely re-align")
    prefix = input_ids[:, :start]
    suffix = input_ids[:, end:]
    middle = torch.full((1, desired_count), sound_token_id, dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat([prefix, middle, suffix], dim=1)


class SingleBatchIterator:
    def __init__(self, input_ids, position_ids, attention_mask, **kwargs):
        self.batch = {
            "tokens": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        self.batch.update({key: value for key, value in kwargs.items() if value is not None})
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def vlm_forward_step(data_iterator, model, **_):
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch["attention_mask"],
        "images": batch["images"],
        "imgs_sizes": batch["imgs_sizes"],
        "num_frames": batch["num_frames"],
        "vision_packed_seq_params": batch["vision_packed_seq_params"],
    }
    if "sound_clips" in batch:
        forward_args["sound_clips"] = batch["sound_clips"]
        forward_args["sound_length"] = batch["sound_length"]
    if getattr(getattr(model, "ddp_config", None), "use_megatron_fsdp", False):
        forward_args["fp32_output"] = False

    def loss_func(output, **__):
        return output

    output = model(**forward_args)
    if isinstance(output, tuple):
        output = output[0]
    return output, loss_func


def _prepare_sample(
    processor,
    tokenizer,
    media_tok,
    question,
    video_path,
    video_fps,
    max_video_frames,
    reasoning,
    include_audio,
    include_video_metadata,
    resize,
):
    image_urls, video_metadata = maybe_path_or_url_to_data_urls(
        video_path,
        fps=max(0, int(video_fps)),
        nframe_max=int(max_video_frames),
    )
    frames = [pil_image_from_base64(image_url) for image_url in image_urls]

    if resize:
        frames = [
                frame.convert("RGB").resize((512, 512))
                for frame in frames
            ]

    messages = [{"role": "user", "content": f"{media_tok}{question}"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=reasoning,
    )

    processor_kwargs = {
        "text": [prompt],
        "videos": frames,
        "return_tensors": "pt",
        "do_sample_frames": False,
    }
    if include_video_metadata:
        processor_kwargs["videos_kwargs"] = {"video_metadata": video_metadata}
    
    if include_audio:
        waveform = load_audio(video_path, target_sr=_AUDIO_SAMPLE_RATE)
        if waveform is None:
            raise ValueError(f'No waveform detected: {video_path}')
        processor_kwargs["audio"] = [waveform]
        processor_kwargs["audio_kwargs"] = {"sampling_rate": _AUDIO_SAMPLE_RATE}
    
    processor_output = processor(**processor_kwargs)

    input_ids = processor_output["input_ids"]

    img_start_id = tokenizer.convert_tokens_to_ids("<img>")
    img_end_id = tokenizer.convert_tokens_to_ids("</img>")
    visual_groups = int((input_ids == img_start_id).sum().item())
    input_ids = adjust_image_tokens(
        input_ids,
        torch.ones(visual_groups, dtype=torch.long),
        img_start_id,
        img_end_id,
    )

    processed_pixels = processor_output["pixel_values_videos"]
    packed_pixels, imgs_sizes = _patchify_video(processed_pixels, _PATCH_DIM)
    num_frames = torch.tensor([len(processed_pixels)], dtype=torch.long)

    sound_clips = sound_length = None
    if include_audio:
        sound_clips = compute_mel_features(
            waveform,
            sampling_rate=_AUDIO_SAMPLE_RATE,
            num_mel_bins=_NUM_MEL_BINS,
        )
        sound_length = torch.tensor([sound_clips.shape[0]], dtype=torch.long)
        sound_clips = sound_clips.unsqueeze(0).bfloat16()

    return {
        "input_ids": input_ids,
        "images": packed_pixels.bfloat16(),
        "imgs_sizes": imgs_sizes,
        "num_frames": num_frames,
        "sound_clips": sound_clips,
        "sound_length": sound_length,
    }, messages


class NemotronOmniMBridge(InferenceModel):
    """Run Nemotron Omni with Megatron model and data parallelism."""

    def __init__(
        self,
        model_path,
        hf_model_path,
        reasoning=False,
        include_audio=True,
        include_video_metadata=True,
        adapter_path=None,
        resize=False,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        is_fsdp=False,
        fsdp_sharding_strategy="no_shard",
        single_line=False,
        **kwargs,
    ):
        if not model_path:
            raise ValueError("model_path must point to a Megatron-Bridge checkpoint")
        if not hf_model_path:
            raise ValueError("hf_model_path is required for the model config and processor")

        peft = None
        if adapter_path:
            run_config = read_run_config(str(Path(adapter_path) / "run_config.yaml"))
            peft = create_peft(run_config["peft"])
            if peft is None:
                raise ValueError("LoRA is not enabled in the adapter run config")

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)

        tp = int(tensor_model_parallel_size)
        ep = int(expert_model_parallel_size)
        etp = int(expert_tensor_parallel_size)

        self.processor = AutoProcessor.from_pretrained(hf_model_path, trust_remote_code=True)

        bridge = AutoBridge.from_hf_pretrained(hf_model_path, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=False)
        provider.tensor_model_parallel_size = tp
        provider.pipeline_model_parallel_size = 1
        provider.context_parallel_size = 1
        provider.expert_model_parallel_size = ep
        provider.expert_tensor_parallel_size = etp
        provider.pipeline_dtype = torch.bfloat16
        provider.dynamic_resolution = True
        provider.temporal_patch_dim = self.processor.video_temporal_patch_dim
        provider.separate_video_embedder = True
        provider.temporal_ckpt_compat = True
        provider.vision_class_token_len = 10
        provider.radio_interpolate_only_cpe = False

        self.is_fsdp = bool(is_fsdp)
        self.fsdp_sharding_strategy = fsdp_sharding_strategy
        if self.is_fsdp:
            _apply_fsdp_overrides()
            self.model = _load_fsdp_model(
                provider,
                model_path,
                self.fsdp_sharding_strategy,
                peft=peft,
                adapter_path=adapter_path,
            )
        else:
            provider.initialize_model_parallel(seed=0)
            self.model = bridge.load_megatron_model(
                model_path,
                mp_overrides={
                    "tensor_model_parallel_size": tp,
                    "pipeline_model_parallel_size": 1,
                    "context_parallel_size": 1,
                    "expert_model_parallel_size": ep,
                    "expert_tensor_parallel_size": etp,
                    "pipeline_dtype": torch.bfloat16,
                    "dynamic_resolution": True,
                    "temporal_patch_dim": self.processor.video_temporal_patch_dim,
                    "separate_video_embedder": True,
                    "temporal_ckpt_compat": True,
                    "vision_class_token_len": 10,
                    "radio_interpolate_only_cpe": False,
                },
                wrap_with_ddp=False,
            )

            if adapter_path:
                peft_model = [model.module if hasattr(model, "module") else model for model in self.model]
                create_peft_hook(peft, training=False)(peft_model)
                load_peft_adapter_checkpoint(peft_model, adapter_path, peft)

        self.model = [model.cuda().eval() for model in self.model]

        unwrapped_model = self.model[0]
        while hasattr(unwrapped_model, "module"):
            unwrapped_model = unwrapped_model.module
        self._sound_output_length = (
            unwrapped_model.llava_model.sound_model.encoder._get_subsampling_output_length
        )

        for model in self.model:
            inner = model.module if hasattr(model, "module") else model
            if hasattr(inner, "config"):
                inner.config.grad_scale_func = None
            if hasattr(inner, "llava_model") and hasattr(inner.llava_model, "config"):
                inner.llava_model.config.grad_scale_func = None

        self._requires_lockstep = (
            (self.is_fsdp and self.fsdp_sharding_strategy != "no_shard")
            or ep > 1
            or etp > 1
        )

        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.reasoning = reasoning
        self.include_audio = include_audio
        self.include_video_metadata = include_video_metadata

        video_tok = self.processor.video_token
        if self.include_audio:
            audio_tok = self.processor.audio_token
            if single_line:
                self.media_tok = f"{video_tok}{audio_tok}"
            else:
                self.media_tok = f"{video_tok}\n{audio_tok}\n"
        else:
            self.media_tok = video_tok

        self.resize = resize

    def inference_distribution(self, _rank, _world_size):
        return InferenceDistribution(
            data_rank=parallel_state.get_data_parallel_rank(),
            data_world_size=parallel_state.get_data_parallel_world_size(),
            writes_results=parallel_state.get_tensor_model_parallel_rank() == 0,
            requires_lockstep=self._requires_lockstep,
        )

    def predict(
        self,
        question,
        video_path,
        video_fps,
        max_video_frames,
        max_new_tokens,
        **kwargs,
    ):
        sample, messages = _prepare_sample(
            self.processor,
            self.tokenizer,
            self.media_tok,
            question,
            video_path,
            video_fps,
            max_video_frames,
            self.reasoning,
            self.include_audio,
            self.include_video_metadata,
            self.resize,
        )
        return self._generate(sample, max_new_tokens), messages

    @torch.inference_mode()
    def _generate(self, sample, max_new_tokens):
        input_ids = sample["input_ids"].to(self.device)
        tensor_inputs = {
            key: value.to(self.device) if value is not None else None
            for key, value in sample.items()
            if key != "input_ids"
        }
        if tensor_inputs["sound_length"] is not None:
            audio_token = getattr(self.tokenizer, "audio_token", "<so_embedding>")
            sound_token_id = self.tokenizer.convert_tokens_to_ids(audio_token)
            expected_sound_tokens = int(
                self._sound_output_length(tensor_inputs["sound_length"]).item()
            )
            input_ids = _align_sound_tokens(input_ids, sound_token_id, expected_sound_tokens)

        prompt_length = input_ids.size(1)
        generated_ids = input_ids.clone()
        stop_token = self.tokenizer.eos_token_id
        stop_length = None
        finished = False
        forward_backward = get_forward_backward_func()

        for _ in range(int(max_new_tokens)):
            position_ids = torch.arange(input_ids.size(1), device=self.device).unsqueeze(0).expand_as(input_ids)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            iterator = SingleBatchIterator(
                input_ids,
                position_ids,
                attention_mask,
                images=tensor_inputs["images"],
                imgs_sizes=tensor_inputs["imgs_sizes"],
                num_frames=tensor_inputs["num_frames"],
                # RADIO mutates this metadata while inserting class tokens.
                vision_packed_seq_params=_build_vision_packed_seq_params(tensor_inputs["imgs_sizes"]),
                sound_clips=tensor_inputs["sound_clips"],
                sound_length=tensor_inputs["sound_length"],
            )

            output = forward_backward(
                forward_step_func=vlm_forward_step,
                data_iterator=iterator,
                model=self.model,
                num_microbatches=1,
                forward_only=True,
                seq_length=input_ids.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )
            if isinstance(output, list) and output:
                output = output[0]
                if isinstance(output, tuple):
                    output = output[0]

            local_logits = output[:, -1:, :].contiguous()
            tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
            if tp_world_size > 1:
                gathered = [torch.empty_like(local_logits) for _ in range(tp_world_size)]
                dist.all_gather(
                    gathered,
                    local_logits,
                    group=parallel_state.get_tensor_model_parallel_group(),
                )
                local_logits = torch.cat(gathered, dim=-1)
            next_token = torch.argmax(local_logits[:, -1], dim=-1, keepdim=True)

            if finished:
                next_token.fill_(stop_token if stop_token is not None else 0)

            generated_ids = torch.cat((generated_ids, next_token), dim=-1)
            input_ids = generated_ids

            if not finished and stop_token is not None and int(next_token.item()) == stop_token:
                finished = True
                stop_length = generated_ids.size(1)

            if self._requires_lockstep:
                all_finished = torch.tensor(int(finished), dtype=torch.int32, device=self.device)
                dist.all_reduce(all_finished, op=dist.ReduceOp.MIN)
                if int(all_finished.item()) == 1:
                    break
            elif finished:
                break

        output_end = stop_length if stop_length is not None else generated_ids.size(1)
        return self.tokenizer.decode(
            generated_ids[0, prompt_length:output_end].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
