# -*- coding: utf-8 -*-
"""Parler-TTS AR decoder, ported off transformers with comfy ``operations``.

State-dict keys are kept 1:1 with the HF checkpoint (``decoder.model.decoder.*``,
``decoder.lm_heads.*``) so ``load_state_dict(strict=True)`` self-validates.

Semantics replicated from parler_tts.modeling_parler_tts (upstream, 4.40 era):
  * 9 codebook embeddings summed into one hidden stream
  * sinusoidal positional embedding (``embed_positions.weights`` is a stored
    parameter, extended on the fly if sequences outgrow the table)
  * pre-norm decoder layer: self-attn (causal) -> cross-attn over the text
    encoder states -> gated FFN, each wrapped in (LayerNorm, residual)
  * per-codebook LM heads stacked to (bsz*9, seq, vocab) logits
Both v1 checkpoints (mini/large) use sinusoidal positions, not RoPE.
"""
import math

import torch
from torch import nn

import comfy.ops


def _layer_norm(dim, dtype, device, operations):
    return operations.LayerNorm(dim, dtype=dtype, device=device)


class ParlerAttention(nn.Module):
    """q/k/v/out_proj without bias, GQA-capable, SDPA based."""

    def __init__(self, embed_dim, num_heads, num_kv_heads, is_causal,
                 dropout=0.0, dtype=None, device=None, operations=comfy.ops.manual_cast):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.is_causal = is_causal

        self.q_proj = operations.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device)
        self.k_proj = operations.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False, dtype=dtype, device=device)
        self.v_proj = operations.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False, dtype=dtype, device=device)
        self.out_proj = operations.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device)

    def _shape_q(self, x, bsz):
        return x.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, x, bsz):
        return x.view(bsz, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states, key_value_states=None, past_kv=None, attn_mask=None):
        """past_kv: in/out tuple (k, v) for caching; None to skip caching.

        key_value_states given -> cross attention (no causal masking of its own).
        attn_mask: additive float mask broadcastable to (bsz, heads, tgt, src).
        """
        is_cross = key_value_states is not None
        bsz, tgt_len = hidden_states.shape[:2]

        # NOTE: HF's SDPA attention class does NOT pre-multiply by self.scaling
        # (only the eager class does); SDPA applies its internal 1/sqrt(d)
        # scale.  Replicate the SDPA path: single scaling only.
        query_states = self._shape_q(self.q_proj(hidden_states), bsz)

        current_states = key_value_states if is_cross else hidden_states
        # keep q/k/v in one dtype no matter where the conditioning came from
        # (e.g. a text encoder that still computes in fp32)
        if current_states.dtype != hidden_states.dtype:
            current_states = current_states.to(hidden_states.dtype)
        new_k = self._shape_kv(self.k_proj(current_states), bsz)
        new_v = self._shape_kv(self.v_proj(current_states), bsz)

        if past_kv is not None:
            if is_cross:
                # cross states are constant over generation: cache on first use
                if past_kv[0] is None:
                    past_kv[0], past_kv[1] = new_k, new_v
                key_states, value_states = past_kv[0], past_kv[1]
            else:
                if past_kv[0] is not None:
                    key_states = torch.cat([past_kv[0], new_k], dim=2)
                    value_states = torch.cat([past_kv[1], new_v], dim=2)
                else:
                    key_states, value_states = new_k, new_v
                past_kv[0], past_kv[1] = key_states, value_states
        else:
            key_states, value_states = new_k, new_v

        if self.num_kv_groups != 1:
            key_states = key_states[:, :, None, :, :].expand(
                bsz, self.num_kv_heads, self.num_kv_groups, key_states.shape[-2], self.head_dim
            ).reshape(bsz, self.num_heads, key_states.shape[-2], self.head_dim)
            value_states = value_states[:, :, None, :, :].expand(
                bsz, self.num_kv_heads, self.num_kv_groups, value_states.shape[-2], self.head_dim
            ).reshape(bsz, self.num_heads, value_states.shape[-2], self.head_dim)

        tgt_for_sdpa = query_states.shape[2]
        is_causal = self.is_causal and not is_cross and attn_mask is None and tgt_for_sdpa > 1
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attn_mask, is_causal=is_causal
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        return self.out_proj(attn_output)


class ParlerDecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx, dtype, device, operations):
        super().__init__()
        hidden = cfg["hidden_size"]
        self.self_attn = ParlerAttention(
            hidden, cfg["num_attention_heads"], cfg["num_key_value_heads"],
            is_causal=True, dropout=cfg.get("attention_dropout", 0.0),
            dtype=dtype, device=device, operations=operations)
        self.encoder_attn = ParlerAttention(
            hidden, cfg["num_attention_heads"], cfg["num_cross_attention_key_value_heads"],
            is_causal=False, dropout=cfg.get("attention_dropout", 0.0),
            dtype=dtype, device=device, operations=operations)
        self.self_attn_layer_norm = _layer_norm(hidden, dtype, device, operations)
        self.encoder_attn_layer_norm = _layer_norm(hidden, dtype, device, operations)
        self.fc1 = operations.Linear(hidden, cfg["ffn_dim"], bias=False, dtype=dtype, device=device)
        self.fc2 = operations.Linear(cfg["ffn_dim"], hidden, bias=False, dtype=dtype, device=device)
        self.final_layer_norm = _layer_norm(hidden, dtype, device, operations)
        self.act_fn = torch.nn.functional.gelu  # gelu (exact); config activation_function="gelu"

    def forward(self, hidden_states, attn_mask, encoder_hidden_states, encoder_attn_mask,
                self_past, cross_past):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, past_kv=self_past, attn_mask=attn_mask)
        hidden_states = residual + hidden_states

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states = self.encoder_attn(
                hidden_states, key_value_states=encoder_hidden_states,
                past_kv=cross_past, attn_mask=encoder_attn_mask)
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.act_fn(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states
        return hidden_states


def _sinusoidal_weights(num_positions, dim, dtype, device):
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, dtype=torch.int64).float() * -emb)
    emb = torch.arange(num_positions, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1).view(num_positions, -1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(num_positions, 1)], dim=1)
    return emb.to(device=device, dtype=dtype)


class ParlerSinusoidalPositionalEmbedding(nn.Module):
    """Weights are a stored parameter (checkpoint key embed_positions.weights);
    extend with a freshly computed table when generation outgrows it."""

    def __init__(self, num_positions, dim, dtype, device):
        super().__init__()
        self.embedding_dim = dim
        self.offset = 2
        # stored as a Parameter: the checkpoint carries it under
        # `embed_positions.weights` (persistent, matches HF)
        self.weights = torch.nn.Parameter(
            _sinusoidal_weights(num_positions, dim, torch.float32, device))
        self.weights.requires_grad_(False)

    def forward(self, seq_len, past_len, device, dtype):
        position_ids = torch.arange(seq_len, device=device) + past_len
        if position_ids[-1] >= self.weights.shape[0]:
            new = _sinusoidal_weights(
                position_ids[-1] + 1 + self.offset, self.embedding_dim, torch.float32, "cpu")
            self.weights.data = new.to(device=self.weights.device)
        return self.weights.detach().index_select(0, position_ids.view(-1)).to(device=device, dtype=dtype)


class ParlerTTSDecoder(nn.Module):
    """Key-compatible with ``decoder.model.decoder.*``."""

    def __init__(self, cfg, dtype, device, operations):
        super().__init__()
        self.num_codebooks = cfg["num_codebooks"]
        self.d_model = cfg["hidden_size"]
        embed_dim = cfg["vocab_size"] + 1  # +1 pad token
        self.embed_tokens = nn.ModuleList([
            operations.Embedding(embed_dim, self.d_model, dtype=dtype, device=device)
            for _ in range(self.num_codebooks)])
        self.embed_positions = ParlerSinusoidalPositionalEmbedding(
            cfg["max_position_embeddings"], self.d_model, dtype, device)
        self.layers = nn.ModuleList([
            ParlerDecoderLayer(cfg, i, dtype, device, operations)
            for i in range(cfg["num_hidden_layers"])])
        self.layer_norm = _layer_norm(self.d_model, dtype, device, operations)

    def forward(self, input_ids, encoder_hidden_states, encoder_attn_mask=None,
                prompt_hidden_states=None, past=None, past_len=0):
        # input_ids: (bsz * num_codebooks, seq)
        codes = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
        bsz, _, seq_len = codes.shape
        inputs_embeds = sum(
            [self.embed_tokens[i](codes[:, i]) for i in range(self.num_codebooks)])

        if prompt_hidden_states is not None:
            inputs_embeds = torch.cat([prompt_hidden_states, inputs_embeds], dim=1)

        total_len = inputs_embeds.shape[1]
        positions = self.embed_positions(total_len, past_len, inputs_embeds.device, inputs_embeds.dtype)
        hidden_states = inputs_embeds + positions

        # Self-attn mask: generation has no decoder padding, so we mirror HF's
        # SDPA fast path exactly -- additive mask stays None and causality is
        # expressed through SDPA's is_causal (only active when query spans >1
        # token; single-token decode steps are unmasked).

        if past is None:
            past = {"self": [[None, None] for _ in self.layers],
                    "cross": [[None, None] for _ in self.layers]}
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, None, encoder_hidden_states, encoder_attn_mask,
                past["self"][i], past["cross"][i])
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states, past


class ParlerTTSModel(nn.Module):
    """Key-compatible with ``decoder.model.*`` (HF ParlerTTSModel wrapper)."""

    def __init__(self, cfg, dtype, device, operations):
        super().__init__()
        self.decoder = ParlerTTSDecoder(cfg, dtype, device, operations)


class ParlerTTSForCausalLM(nn.Module):
    """Key-compatible with ``decoder.*`` (model + lm_heads)."""

    def __init__(self, cfg, dtype, device, operations):
        super().__init__()
        self.model = ParlerTTSModel(cfg, dtype, device, operations)
        self.num_codebooks = cfg["num_codebooks"]
        self.vocab_size = cfg["vocab_size"]
        self.lm_heads = nn.ModuleList([
            operations.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False, dtype=dtype, device=device)
            for _ in range(self.num_codebooks)])

    def forward(self, input_ids, encoder_hidden_states, encoder_attn_mask=None,
                prompt_hidden_states=None, past=None, past_len=0):
        hidden_states, past = self.model.decoder(
            input_ids, encoder_hidden_states, encoder_attn_mask,
            prompt_hidden_states, past, past_len)
        logits = torch.stack([head(hidden_states) for head in self.lm_heads], dim=1)
        # (bsz, num_codebooks, seq, vocab) -> (bsz * num_codebooks, seq, vocab)
        logits = logits.reshape(-1, *logits.shape[2:])
        return logits, past
