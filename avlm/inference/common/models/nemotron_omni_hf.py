import os
import json
from pathlib import Path
from safetensors import safe_open
import re
import time

import torch
from transformers import (AutoConfig, 
                          #AutoModelForCausalLM, 
                          AutoModel,
                          AutoProcessor, 
                          #AutoTokenizer,
                          )

from avlm.inference.common.models import InferenceModel
from avlm.inference.common.utils.video_io import (
    maybe_path_or_url_to_data_urls,
    pil_image_from_base64,
)

PROCESSOR_METADATA_KEYS = ("num_patches", "num_tokens", "imgs_sizes")
DIST_ENV_VARS = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")

# Wrapper -> HF base FQN translation. vision_projector.* targets are listed in
# adapter_config.json but no tensors are saved for them, so we just skip those.
def translate(fqn):
    if fqn.startswith("language_model.model."):
        return "language_model.backbone." + fqn[len("language_model.model."):]
    return fqn
    #return None

def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default

    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False

    raise ValueError(f"{name} must be boolean-like, got {os.environ[name]!r}")


# will fail if audio is included and
# OMP_NUM_THREADS=1 is not set
class NemotronOmniHF(InferenceModel):
    def __init__(self, 
                 model_path,  
                 hf_model_path,
                 reasoning=False, 
                 include_audio=True,
                 include_video_metadata=True,
                 adapter_path=None,
                 single_line=False,
                 resize=False,
                 **kwargs):
        
        if not model_path:
            raise ValueError('Model path must be set')

        consolidated_path = os.path.join(model_path, 'model', 'consolidated')
        if os.path.exists(consolidated_path):
            model_path = consolidated_path

        if (adapter_path) and ('consolidated' in model_path):
            raise ValueError(f'LoRA should used based model, not: {model_path}')

        if adapter_path:
            adapter_model_path = os.path.join(adapter_path, 'model')
            if os.path.exists(adapter_model_path):
                adapter_path = adapter_model_path

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")

        local_files_only = env_bool("LOCAL_FILES_ONLY", False)

        time.sleep(local_rank * 5)

        env_backup = {}
        for var in DIST_ENV_VARS:
            if var in os.environ:
                env_backup[var] = os.environ.pop(var)
        try:

            ###
            # Transformers incorrectly stages only direct imports for local `trust_remote_code` models,
            # omitting transitive files such as configuration_radio.py from HF_MODULES_CACHE.
            # Use hf_model_path for model code while loading the converted config and weights from model_path.
            base_config = AutoConfig.from_pretrained(
                hf_model_path,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )

            config = type(base_config).from_pretrained(model_path, 
                                                trust_remote_code=True, 
                                                local_files_only=local_files_only,
                                                )
            
            config.auto_map["AutoModel"] = (
                f"{hf_model_path}--{base_config.auto_map['AutoModel']}"
            )

            # for when bug is fixed
            # config = AutoConfig.from_pretrained(model_path, 
            #                                     trust_remote_code=True, 
            #                                     local_files_only=local_files_only,
            #                                     )

            ###

            config.video_pruning_rate = 0.0
            # not sure if this or Automodel
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=None,
                #device_map={"": self.device},
                config=config,
                low_cpu_mem_usage=True,
                local_files_only=local_files_only,
            )

            # not sure if better to load or use
            # tokenizer = self.processor.tokenizer
            # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_files_only)

            # LoRA loads the processor from the base model, which carries the stock video
            # resolution. Take the resolution the adapter was actually trained at instead.
            if adapter_path:
                adapter_processor_config = os.path.join(adapter_path, 'processor_config.json')
                if os.path.exists(adapter_processor_config):
                    with open(adapter_processor_config) as f:
                        trained_num_patches = json.load(f).get('image_processor', {}).get('video_target_num_patches')
                    if trained_num_patches:
                        self.processor.image_processor.video_target_num_patches = trained_num_patches

            self.tokenizer = self.processor.tokenizer

            # necessary else device side assert errors
            if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "radio_model"):
                self.model.vision_model.radio_model.summary_idxs = None
            
            self.model.to(self.device)

            ### lora
            if adapter_path:
                cfg   = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
                scale = cfg["lora_alpha"] / cfg["r"]
                pairs = {}
                with safe_open(str(Path(adapter_path) / "adapter_model.safetensors"), framework="pt") as f:
                    for k in f.keys():
                        m = re.match(r"^base_model\.model\.(.+)\.lora_(A|B)\.weight$", k)
                        if m:
                            pairs.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(k)

                modules = dict(self.model.named_modules())
                for wrapper_fqn, ab in pairs.items():
                    hf_fqn = translate(wrapper_fqn)
                    if hf_fqn is None or hf_fqn not in modules:
                        continue
                    W = modules[hf_fqn].weight
                    A = ab["A"].to(device=W.device, dtype=torch.float32)
                    B = ab["B"].to(device=W.device, dtype=torch.float32)
                    with torch.no_grad():
                        # This rounds the LoRA delta before BF16 addition, so it can differ slightly from full export.
                        # To match full export, use: W.copy_((W.float() + (B @ A) * scale).to(W.dtype))
                        W.add_(((B @ A) * scale).to(W.dtype))
            ###

            self.model.eval()
        finally:
            for var, value in env_backup.items():
                os.environ[var] = value

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
            self.media_tok=video_tok

        self.resize = resize

    def predict(self, 
                question, 
                video_path,
                video_fps,
                max_video_frames,
                max_new_tokens,
                audio_sampling_rate=16000,
                repetition_penalty=None,
                no_repeat_ngram_size=None,
                **kwargs,
                ):

        image_urls, video_metadata = maybe_path_or_url_to_data_urls(
            video_path,
            fps=max(0, int(video_fps)),
            nframe_max=int(max_video_frames),
        )
        frames = [pil_image_from_base64(image_url) for image_url in image_urls]

        if self.resize:
            frames = [
                frame.convert("RGB").resize((512, 512))
                for frame in frames
            ]
        
        messages = [
            #{"role": "system", "content": "Answer the questions."},
            {"role": "user", "content": f"{self.media_tok}{question}"},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.reasoning,
        )

        processor_kwargs = {
            'text': [prompt],
            'videos': frames,
            'return_tensors': 'pt',
            'do_sample_frames': False,
        }

        if self.include_video_metadata:
            processor_kwargs['videos_kwargs'] = {'video_metadata': video_metadata}

        if self.include_audio:
            processor_kwargs['audio'] = [video_path]
            processor_kwargs['audio_kwargs'] = {"sampling_rate": audio_sampling_rate}

        inputs = self.processor(**processor_kwargs)

        for k in PROCESSOR_METADATA_KEYS:
            inputs.pop(k, None)

        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()}
        
        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=False, # needs this or commented line below
            output_hidden_states=False, # unsure if necessary
            #eos_token_id=self.tokenizer.eos_token_id,
        )
        if repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size is not None:
            generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        # if not isinstance(generated_ids, torch.Tensor):
        #     generated_ids = generated_ids.sequences

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0].strip(), messages       
