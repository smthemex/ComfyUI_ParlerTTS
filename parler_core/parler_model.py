# -*- coding: utf-8 -*-
"""Composite Parler-TTS model + native comfy loader.

Module attribute names mirror the checkpoint's top-level prefixes exactly
(``text_encoder.*``, ``decoder.*``, ``audio_encoder.*``, ``embed_prompts``,
``enc_to_dec_proj``) so the full safetensors loads with ``strict=True`` and
self-validates every key.  Text encoder reuses comfy's T5 core; the decoder
and DAC are local ports with zero transformers/diffusers/audiotools imports.
"""
import torch
from torch import nn

import comfy.model_management
import comfy.model_patcher
import comfy.ops

from .decoder_native import ParlerTTSForCausalLM
from .dac_native import DACModelLite
from .t5_native import T5TextEncoder
from . import builtin
from . import generate as gen


class ParlerTTSComposite(nn.Module):
    """Keys 1:1 with the checkpoint.  ``strict=True`` load is the contract."""

    def __init__(self, config: dict, operations=comfy.ops.manual_cast):
        super().__init__()
        te_cfg = config["text_encoder"]
        dec_cfg = config["decoder"]
        self.dec_cfg = dec_cfg
        self.text_encoder = T5TextEncoder(te_cfg, dtype=None, device="cpu", operations=operations)
        self.decoder = ParlerTTSForCausalLM(dec_cfg, dtype=None, device="cpu", operations=operations)
        self.audio_encoder = DACModelLite(config["audio_encoder"])
        self.embed_prompts = operations.Embedding(
            config["vocab_size"], dec_cfg["hidden_size"], device="cpu")
        if te_cfg["d_model"] != dec_cfg["hidden_size"]:
            self.enc_to_dec_proj = operations.Linear(
                te_cfg["d_model"], dec_cfg["hidden_size"], bias=True, device="cpu")

    def forward(self, *args, **kwargs):  # not used; nodes call the pipeline
        raise NotImplementedError


class ParlerTTSPipeline:
    """Owns the ModelPatcher and provides the TTS entry point."""

    def __init__(self, ckpt_path: str, preset: str = "auto", dtype=None):
        """ckpt_path: a single safetensors checkpoint (or a safetensors index /
        shard for large-v1).  Configs and tokenizer come from the plugin's
        built-in configs/ directory, never from the model folder."""
        self.ckpt_path = ckpt_path
        sd = builtin.load_checkpoint(ckpt_path)
        self.preset = builtin.detect_preset(sd) if preset == "auto" else preset
        self.config = builtin.load_config(self.preset)
        self.generation_config = builtin.load_generation_config(self.preset)
        builtin.validate_config(self.config, sd, self.preset)
        self.sampling_rate = self.config["audio_encoder"]["sampling_rate"]
        self.codebook_size = self.config["audio_encoder"]["codebook_size"]

        if dtype is not None:
            sd = {k: (v.to(dtype) if k.startswith(
                ("text_encoder.", "decoder.", "embed_prompts", "enc_to_dec_proj.")) else v)
                for k, v in sd.items()}

        self.model = ParlerTTSComposite(self.config)
        missing, unexpected = self.model.load_state_dict(sd, strict=True, assign=True)
        self.model.requires_grad_(False)
        self.model.eval()

        load_device = comfy.model_management.get_torch_device()
        offload_device = torch.device("cpu")
        self.patcher = comfy.model_patcher.ModelPatcher(
            self.model, load_device=load_device, offload_device=offload_device)
        comfy.model_management.load_models_gpu([self.patcher], force_full_load=True)

    # ------------------------------------------------------------------
    @property
    def compute_dtype(self):
        """The decoder's weight dtype is the single source of truth for the
        AR path (fp32 or bf16).  Conditioning tensors are aligned to it."""
        return self.model.decoder.lm_heads[0].weight.dtype

    @torch.no_grad()
    def encode_description(self, input_ids, attention_mask):
        model = self.model
        hidden = model.text_encoder(input_ids, attention_mask)
        if hasattr(model, "enc_to_dec_proj"):
            hidden = model.enc_to_dec_proj(hidden)
        hidden = hidden * attention_mask[..., None].to(hidden.dtype)
        return hidden.to(self.compute_dtype)

    @staticmethod
    def build_encoder_attn_mask(attention_mask, dtype):
        """(bsz, seq) 0/1 mask -> additive 4D mask; all-ones -> None (SDPA fast
        path, matching HF's _prepare_4d_attention_mask_for_sdpa)."""
        if attention_mask is None or bool((attention_mask == 1).all()):
            return None
        min_dtype = torch.finfo(dtype).min
        mask = (1.0 - attention_mask.to(dtype)) * min_dtype
        return mask[:, None, None, :]

    @torch.no_grad()
    def tts(self, input_ids, attention_mask, prompt_input_ids=None, *,
            seed=0, do_sample=True, temperature=1.0, top_k=50, top_p=1.0,
            max_length=2580, min_new_tokens=10, progress_cb=None):
        """input_ids: (1, L) description tokens; prompt_input_ids: optional
        voice-prompt tokens.  Returns audio waveform (1, 1, samples) in [-1, 1]."""
        encoder_hidden = self.encode_description(input_ids, attention_mask)
        encoder_attn_mask = self.build_encoder_attn_mask(attention_mask, encoder_hidden.dtype)

        prompt_hidden = None
        if prompt_input_ids is not None:
            prompt_hidden = self.model.embed_prompts(prompt_input_ids).to(self.compute_dtype)

        dec_cfg = self.config["decoder"]
        gen_cfg = self.generation_config
        codes = gen.generate_codes(
            self.model.decoder, dec_cfg["num_codebooks"], encoder_hidden, encoder_attn_mask,
            prompt_hidden_states=prompt_hidden,
            seed=seed, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
            max_length=max_length, min_new_tokens=min_new_tokens,
            bos_token_id=gen_cfg.get("decoder_start_token_id", self.config["decoder_start_token_id"]),
            pad_token_id=gen_cfg.get("pad_token_id", self.config["pad_token_id"]),
            eos_token_id=gen_cfg.get("eos_token_id", dec_cfg["eos_token_id"]),
            device=encoder_hidden.device,
            progress_cb=progress_cb)

        trimmed = gen.trim_codes(codes, self.codebook_size)
        audios = [self.model.audio_encoder.decode_codes(c[None, ...])[0] for c in trimmed]
        max_len = max(a.shape[-1] for a in audios)
        out = torch.zeros((len(audios), 1, max_len), dtype=audios[0].dtype, device=audios[0].device)
        for i, a in enumerate(audios):
            out[i, 0, : a.shape[-1]] = a[0]
        return out
