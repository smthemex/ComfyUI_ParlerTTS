# -*- coding: utf-8 -*-
"""Built-in model configs, shared tokenizer and single-checkpoint loading.

Nothing is read from the model folder any more: a Parler-TTS checkpoint is a
single self-contained safetensors file (or a sharded safetensors index for
large-v1), while configs / generation defaults / tokenizer files live in the
plugin's ``configs/`` directory.

Preset detection is driven by checkpoint evidence (decoder hidden size,
enc_to_dec_proj, layer count) -- see the porting rule "structure is defined by
the checkpoint, not by config defaults".
"""
import json
import os

import comfy.utils

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
TOKENIZER_DIR = os.path.join(CONFIG_DIR, "tokenizer")
PRESETS = ("mini_v1", "large_v1")


def load_config(preset: str) -> dict:
    path = os.path.join(CONFIG_DIR, preset, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"built-in config missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_generation_config(preset: str) -> dict:
    path = os.path.join(CONFIG_DIR, preset, "generation_config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_preset(sd: dict) -> str:
    """Infer mini_v1 / large_v1 from the checkpoint's own tensors."""
    lm = sd.get("decoder.lm_heads.0.weight")
    if lm is None:
        raise ValueError("checkpoint has no decoder.lm_heads.0.weight: not a Parler-TTS model")
    hidden = lm.shape[1]
    has_proj = any(k.startswith("enc_to_dec_proj.") for k in sd)
    layers = sum(1 for k in sd if k.startswith("decoder.model.decoder.layers.") and k.endswith(".fc1.weight"))
    if has_proj or hidden == 1536:
        return "large_v1"
    if hidden == 1024:
        return "mini_v1"
    raise ValueError(
        f"unknown Parler-TTS architecture (decoder hidden={hidden}, layers={layers}, "
        f"enc_to_dec_proj={has_proj}); pick a built-in preset explicitly")


def validate_config(config: dict, sd: dict, preset: str) -> None:
    """Cheap shape checks before the strict load produces a 1000-line error."""
    dec = config["decoder"]
    te = config["text_encoder"]
    checks = [
        ("decoder.hidden_size", dec["hidden_size"], sd["decoder.lm_heads.0.weight"].shape[1]),
        ("decoder.ffn_dim", dec["ffn_dim"], sd["decoder.model.decoder.layers.0.fc1.weight"].shape[0]),
        ("decoder.vocab_size", dec["vocab_size"], sd["decoder.lm_heads.0.weight"].shape[0]),
        ("text_encoder.d_model", te["d_model"], sd["text_encoder.shared.weight"].shape[1]),
        ("text_encoder.d_ff", te["d_ff"], sd["text_encoder.encoder.block.0.layer.1.DenseReluDense.wi_0.weight"].shape[0]),
        ("text_encoder.vocab_size", te["vocab_size"], sd["text_encoder.shared.weight"].shape[0]),
    ]
    n_layers = sum(1 for k in sd if k.startswith("decoder.model.decoder.layers.") and k.endswith(".fc1.weight"))
    checks.append(("decoder.num_hidden_layers", dec["num_hidden_layers"], n_layers))
    bad = [(name, a, b) for name, a, b in checks if a != b]
    if bad:
        raise ValueError(
            f"checkpoint does not match built-in preset '{preset}': {bad}. "
            "Put the matching config.json into configs/<preset>/ or select another preset.")


def load_checkpoint(ckpt_path: str) -> dict:
    """Single safetensors file, safetensors index, or one of its shards."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    lower = ckpt_path.lower()

    if lower.endswith(".json"):
        return _load_index(ckpt_path)

    if not lower.endswith((".safetensors", ".sft")):
        raise ValueError(f"unsupported checkpoint file: {ckpt_path}")

    # a shard selected directly: prefer the sibling index so all weights load
    folder = os.path.dirname(ckpt_path)
    index_path = os.path.join(folder, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            shards = set(json.load(f)["weight_map"].values())
        if os.path.basename(ckpt_path) in shards:
            return _load_index(index_path)

    return comfy.utils.load_torch_file(ckpt_path)


def _load_index(index_path: str) -> dict:
    folder = os.path.dirname(index_path)
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))
    missing = [s for s in shards if not os.path.isfile(os.path.join(folder, s))]
    if missing:
        raise FileNotFoundError(
            f"sharded checkpoint incomplete, missing {len(missing)}/{len(shards)} files: {missing[:3]}")
    sd = {}
    for shard in shards:
        sd.update(comfy.utils.load_torch_file(os.path.join(folder, shard)))
    return sd
