# -*- coding: utf-8 -*-
"""Delay-pattern handling and the autoregressive sampling loop.

Replicates upstream parler_tts.modeling_parler_tts + transformers 4.40
``GenerationMixin._sample`` semantics for the single-beam sample path:

  * ``build_delay_pattern_mask`` / ``apply_delay_pattern_mask``: verbatim ports
  * logits processors: ParlerTTS EOS cascade, MinLength (EOS ban), then warpers
    temperature -> top_k -> top_p (HF order; GenerationConfig defaults make
    top_k=50 active even at temperature=1.0)
  * multinomial sampling over softmax probs, one draw per (bsz*num_codebooks) row
  * finished rows are forced to pad (= eos = 1024) and stop when every row's
    last token is eos (EosTokenCriteria) or max_length is reached
"""
import math

import torch


# ---------------------------------------------------------------------------
# delay pattern (verbatim port from upstream)
# ---------------------------------------------------------------------------

def build_delay_pattern_mask(input_ids, bos_token_id, pad_token_id, max_length, num_codebooks):
    """See upstream docstring: codebook c is offset by c steps, BOS fills the
    leading triangle and PAD the trailing triangle; -1 marks free positions."""
    input_ids = input_ids.reshape(-1, num_codebooks, input_ids.shape[-1])
    bsz, num_codebooks, seq_len = input_ids.shape

    input_ids_shifted = torch.ones((bsz, num_codebooks, max_length), dtype=torch.long,
                                   device=input_ids.device) * -1

    if max_length < 2 * num_codebooks - 1:
        return input_ids.reshape(bsz * num_codebooks, -1), input_ids_shifted.reshape(bsz * num_codebooks, -1)

    for codebook in range(num_codebooks):
        input_ids_shifted[:, codebook, codebook: seq_len + codebook] = input_ids[:, codebook]

    eos_delay_pattern = torch.triu(
        torch.ones((num_codebooks, max_length), dtype=torch.bool),
        diagonal=max_length - num_codebooks + 1)
    bos_delay_pattern = torch.tril(torch.ones((num_codebooks, max_length), dtype=torch.bool))

    bos_mask = ~(bos_delay_pattern).to(input_ids.device)
    eos_mask = ~(eos_delay_pattern).to(input_ids.device)
    mask = ~(bos_delay_pattern + eos_delay_pattern).to(input_ids.device)
    input_ids = mask * input_ids_shifted + ~bos_mask * bos_token_id + ~eos_mask * pad_token_id

    first_codebook_ids = input_ids[:, 0, :]
    start_ids = (first_codebook_ids == -1).nonzero()[:, 1]
    first_start_id = min(start_ids) if len(start_ids) > 0 else seq_len

    pattern_mask = input_ids.reshape(bsz * num_codebooks, -1)
    input_ids = input_ids[..., :first_start_id].reshape(bsz * num_codebooks, -1)
    return input_ids, pattern_mask


def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
    seq_len = input_ids.shape[-1]
    decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
    input_ids = torch.where(decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask)
    return input_ids


# ---------------------------------------------------------------------------
# logits processors (HF 4.40 semantics)
# ---------------------------------------------------------------------------

class ParlerEOSCascadeProcessor:
    """Upstream ParlerTTSLogitsProcessor: codebooks above the first unfinished
    one may never emit EOS (the EOS cascade must propagate codebook by codebook)."""

    def __init__(self, eos_token_id, num_codebooks, batch_size, device):
        self.eos_token_id = int(eos_token_id)
        self.num_codebooks = num_codebooks
        self.batch_size = batch_size
        self.codebook_idx = torch.arange(batch_size * num_codebooks, device=device)
        self.first_codebooks_unfinished = torch.arange(batch_size, device=device) * num_codebooks
        self.max_codebooks = (torch.arange(batch_size, device=device) * num_codebooks
                              + num_codebooks - 1)

    def __call__(self, input_ids, scores):
        is_eos = (input_ids == self.eos_token_id).sum(1)
        self.first_codebooks_unfinished = torch.where(
            (is_eos[self.first_codebooks_unfinished] > 0)
            & (self.first_codebooks_unfinished < self.max_codebooks),
            self.first_codebooks_unfinished + 1,
            self.first_codebooks_unfinished)
        eos_token_mask = self.codebook_idx > self.first_codebooks_unfinished.repeat_interleave(self.num_codebooks)
        scores[eos_token_mask, self.eos_token_id] = -math.inf
        return scores


class MinLengthProcessor:
    def __init__(self, min_length, eos_token_id):
        self.min_length = min_length
        self.eos_token_id = int(eos_token_id)

    def __call__(self, input_ids, scores):
        if input_ids.shape[-1] < self.min_length:
            scores[:, self.eos_token_id] = -math.inf
        return scores


class TopKWarper:
    def __init__(self, top_k, min_tokens_to_keep=1):
        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = -float("inf")

    def __call__(self, input_ids, scores):
        top_k = min(self.top_k, scores.size(-1))
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        return scores.masked_fill(indices_to_remove, self.filter_value)


class TopPWarper:
    def __init__(self, top_p, min_tokens_to_keep=1):
        self.top_p = top_p
        self.min_tokens_to_keep = min_tokens_to_keep
        self.filter_value = -float("inf")

    def __call__(self, input_ids, scores):
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove)
        return scores.masked_fill(indices_to_remove, self.filter_value)


# ---------------------------------------------------------------------------
# sampling loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_codes(decoder, num_codebooks, encoder_hidden, encoder_mask_4d,
                   prompt_hidden_states=None, *, seed=0, do_sample=True,
                   temperature=1.0, top_k=50, top_p=1.0, max_length=2580,
                   min_new_tokens=10, bos_token_id=1025, pad_token_id=1024,
                   eos_token_id=1024, device=None, progress_cb=None):
    """Returns codes of shape (bsz, num_codebooks, T) with delay pattern
    reverted, values in [0, vocab_size); contains bos/pad/eos markers."""
    bsz = encoder_hidden.shape[0]
    rows = bsz * num_codebooks
    device = device if device is not None else encoder_hidden.device
    torch.manual_seed(seed)

    input_ids = torch.ones((rows, 1), dtype=torch.long, device=device) * bos_token_id
    input_ids, pattern_mask = build_delay_pattern_mask(
        input_ids, bos_token_id, pad_token_id, max_length, num_codebooks)

    # HF: min_length = min_new_tokens + input_ids_length (decoder input is 1 col)
    processors = []
    cascade = ParlerEOSCascadeProcessor(eos_token_id, num_codebooks, bsz, device)
    processors.append(cascade)
    if min_new_tokens is not None and min_new_tokens > 0:
        processors.append(MinLengthProcessor(min_new_tokens + 1, eos_token_id))
    warpers = []
    if temperature is not None and temperature != 1.0:
        warpers.append(lambda ids, s: s / temperature)
    if top_k is not None and top_k != 0:
        warpers.append(TopKWarper(top_k))
    if top_p is not None and top_p < 1.0:
        warpers.append(TopPWarper(top_p))

    unfinished = torch.ones(rows, dtype=torch.long, device=device)
    cur_len = input_ids.shape[-1]
    past = None
    past_len = 0
    first_step = True

    while unfinished.max() > 0 and cur_len < max_length:
        # HF prepare_inputs_for_generation: pattern is applied to the raw token
        # stream each step (bos/pad triangles are filler), then only the last
        # column is fed when a KV cache exists.
        step_full = apply_delay_pattern_mask(input_ids, pattern_mask)
        step_ids = step_full if past is None else step_full[:, -1:]
        logits, past = decoder(
            step_ids,
            encoder_hidden_states=encoder_hidden,
            encoder_attn_mask=encoder_mask_4d,
            prompt_hidden_states=prompt_hidden_states if first_step else None,
            past=past,
            past_len=past_len,
        )
        first_step = False
        past_len = past["self"][0][0].shape[2] if past["self"][0][0] is not None else past_len

        next_token_logits = logits[:, -1, :]
        for proc in processors:
            next_token_logits = proc(input_ids, next_token_logits)
        for warp in warpers:
            next_token_logits = warp(input_ids, next_token_logits)

        if do_sample:
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)

        next_tokens = next_tokens * unfinished + pad_token_id * (1 - unfinished)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        unfinished = unfinished & ~((input_ids[:, -1] == eos_token_id))
        cur_len += 1
        if progress_cb is not None:
            progress_cb(cur_len)

    output_ids = apply_delay_pattern_mask(input_ids, pattern_mask)

    # revert the delay pattern: drop bos / pad filler positions
    _, mask = build_delay_pattern_mask(
        torch.ones((rows, 1), dtype=torch.long, device=device) * bos_token_id,
        bos_token_id, pad_token_id, output_ids.shape[1], num_codebooks)
    mask = (mask != bos_token_id) & (mask != pad_token_id)
    output_ids = output_ids[mask].reshape(bsz, num_codebooks, -1)
    return output_ids


@torch.no_grad()
def trim_codes(codes, codebook_size):
    """Replicates the generate() decode path: keep only columns where every
    codebook holds a real code (< codebook_size)."""
    valid = (codes < codebook_size).all(dim=1)  # (bsz, T)
    out = []
    for b in range(codes.shape[0]):
        out.append(codes[b][:, valid[b]])
    return out
