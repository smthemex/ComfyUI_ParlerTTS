# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Parler-TTS nodes on comfy's native inference stack.

ParlerTTS_LoadModel : comfy-native single-checkpoint dropdown -> PARLERTTS_MODEL
ParlerTTS_Sampler  : description + optional voice prompt -> native AUDIO
                     (feed into comfy's built-in SaveAudio / PreviewAudio)

The model is a SINGLE self-contained safetensors file (or a sharded index for
large-v1) placed in ``models/parler_tts/``.  Everything else -- config.json,
generation_config.json and the flan-t5 tokenizer -- is built into the plugin's
``configs/`` folder, so the model folder itself needs no sidecar files.

Zero transformers model code: the text encoder reuses comfy's T5 core, the
AR decoder and DAC codec are torch-only ports (parler_core/).  The only
third-party dependency left is transformers' AutoTokenizer (stable API).
"""
import os

import torch

import comfy.model_management
import comfy.utils
import folder_paths

from .parler_core import builtin
from .parler_core.parler_model import ParlerTTSPipeline


def _get_tokenizer():
    # lazy import: transformers is only needed for tokenization
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(builtin.TOKENIZER_DIR)


class ParlerTTS_LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("clip"),),
                "preset": (["auto", *builtin.PRESETS], {"default": "auto",
                           "tooltip": "auto detects mini/large from checkpoint tensors"}),
                "dtype": (["fp32", "bf16"], {"default": "fp32"}),
            }
        }

    RETURN_TYPES = ("PARLERTTS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "main_loader"
    CATEGORY = "Parler_TTS"
    DESCRIPTION = ("Load one Parler-TTS safetensors checkpoint. Configs and tokenizer "
                   "are built into the plugin; put the .safetensors in models/clip/.")

    def main_loader(self, ckpt_name, preset, dtype):
        ckpt_path = folder_paths.get_full_path_or_raise("clip", ckpt_name)
        torch_dtype = torch.float32 if dtype == "fp32" else torch.bfloat16
        pipeline = ParlerTTSPipeline(ckpt_path, preset=preset, dtype=torch_dtype)
        return ({
            "pipeline": pipeline,
            "preset": pipeline.preset,
            "gen_config": pipeline.generation_config,
        },)


class ParlerTTS_Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PARLERTTS_MODEL",),
                "description": ("STRING", {"multiline": True, "default": "A female speaker with a slightly low-pitched voice delivers her words "
                 "quite expressively, in a very confined sounding environment with clear "
                 "audio quality. She speaks very fast."}),
                "voice_prompt": ("STRING", {"multiline": True, "default": "Hey, how are you doing today?",
                                            "tooltip": "Optional continuation text; empty = description only"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 1024,
                                  "tooltip": "0 disables top-k (0 != HF default; HF applies 50 by default)"}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_length": ("INT", {"default": -1, "min": -1, "max": 4096,
                                       "tooltip": "-1 = use built-in generation_config (2580 frames, ~30s)"}),
                "min_new_tokens": ("INT", {"default": -1, "min": -1, "max": 512,
                                           "tooltip": "-1 = use built-in generation_config (mini 10, large 0)"}),
                "do_sample": (["enable", "disable"], {"default": "enable"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "prompt2audio"
    CATEGORY = "Parler_TTS"

    def prompt2audio(self, model, description, voice_prompt, seed, temperature,
                     top_k, top_p, max_length, min_new_tokens, do_sample):
        pipeline = model["pipeline"]
        gen_cfg = model["gen_config"]
        max_length = gen_cfg.get("max_length", 2580) if max_length < 0 else max_length
        min_new_tokens = gen_cfg.get("min_new_tokens", 0) if min_new_tokens < 0 else min_new_tokens

        tokenizer = _get_tokenizer()
        device = comfy.model_management.get_torch_device()

        desc_enc = tokenizer(description, return_tensors="pt")
        input_ids = desc_enc.input_ids.to(device)
        attention_mask = desc_enc.attention_mask.to(device)

        prompt_ids = None
        if voice_prompt and voice_prompt.strip():
            prompt_ids = tokenizer(voice_prompt, return_tensors="pt").input_ids.to(device)

        pbar = comfy.utils.ProgressBar(min(max_length, 2580))
        audio = pipeline.tts(
            input_ids, attention_mask, prompt_ids,
            seed=seed, do_sample=(do_sample == "enable"),
            temperature=temperature, top_k=top_k, top_p=top_p,
            max_length=max_length, min_new_tokens=min_new_tokens,
            progress_cb=lambda step: pbar.update(1),
        )

        audio_out = {
            "waveform": audio.cpu(),
            "sample_rate": pipeline.sampling_rate,
        }
        return (audio_out,)


NODE_CLASS_MAPPINGS = {
    "ParlerTTS_LoadModel": ParlerTTS_LoadModel,
    "ParlerTTS_Sampler": ParlerTTS_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParlerTTS_LoadModel": "ParlerTTS Load Checkpoint (Native)",
    "ParlerTTS_Sampler": "ParlerTTS Sampler (Native)",
}
