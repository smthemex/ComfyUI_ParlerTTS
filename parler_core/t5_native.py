# -*- coding: utf-8 -*-
"""Parler-TTS text encoder on top of comfy's native T5 core.

Reuses ``comfy.text_encoders.t5.T5`` (config-driven, key-compatible with HF T5)
so the checkpoint's ``text_encoder.*`` keys load 1:1.  No transformers import.
"""
import torch

import comfy.ops
from comfy.text_encoders import t5 as comfy_t5


# flan-t5 uses "gelu_new" (= tanh-approximated gelu); comfy's table calls it
# "gelu_pytorch_tanh".  Map every known alias to comfy's activation names.
_ACT_MAP = {
    "gelu_new": "gelu_pytorch_tanh",
    "gelu_pytorch_tanh": "gelu_pytorch_tanh",
    "relu": "relu",
}


def build_t5_config(te_cfg: dict) -> dict:
    """te_cfg: the ``text_encoder`` section of ParlerTTS config.json."""
    return {
        "num_layers": te_cfg["num_layers"],
        "d_model": te_cfg["d_model"],
        "d_kv": te_cfg["d_kv"],
        "num_heads": te_cfg["num_heads"],
        "d_ff": te_cfg["d_ff"],
        "dense_act_fn": _ACT_MAP[te_cfg.get("dense_act_fn", "gelu_new")],
        "is_gated_act": te_cfg.get("is_gated_act", True),
        "num_heads": te_cfg["num_heads"],
        "vocab_size": te_cfg["vocab_size"],
        "model_type": te_cfg.get("model_type", "t5"),
    }


class T5TextEncoder(comfy_t5.T5):
    """Subclasses comfy's T5 directly so state-dict keys stay
    ``text_encoder.shared.*`` / ``text_encoder.encoder.*`` (checkpoint names);
    forward returns the hidden-state tensor instead of the (x, intermediate)
    tuple comfy's core returns."""

    def __init__(self, te_cfg: dict, dtype=None, device="cpu", operations=comfy.ops.manual_cast):
        super().__init__(build_t5_config(te_cfg), dtype, device, operations)

    def forward(self, input_ids, attention_mask=None):
        # comfy's T5 defaults to out_dtype=float32 for the embedding, which
        # would make the whole encoder output fp32 even when its weights are
        # bf16.  The decoder/cross-attention then sees a q(bf16)/kv(fp32)
        # mismatch.  Force the encoder to compute in its own storage dtype.
        out = super().forward(
            input_ids, attention_mask=attention_mask, dtype=self.shared.weight.dtype)
        return out[0] if isinstance(out, tuple) else out
