# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in HF-Qwen2 talker mode for CosyVoice3 (bf16-capable).

The default CosyVoice3 talker is the custom ``CosyVoice3LM`` (separate
``speech_embedding`` / ``llm_decoder``), which only runs correctly in fp32.
Setting ``COSYVOICE3_HF_TALKER=1`` swaps the talker (stage 0) to the folded
vanilla Qwen2 produced by ``convert_cosyvoice3_to_hf.py``
(``hf_cosyvoice3_llm``), which is bf16-stable: ``speech_embedding`` is folded
into ``embed_tokens`` and ``llm_decoder`` into ``lm_head`` (text rows masked
to -inf via the lm_head bias), so no runtime logsumexp / embed-rearrange is
needed and stop is a single eos token.

This module is intentionally dependency-free (only ``os``/``json``) so it can
be imported from the config, the pipeline topology, and the model without
creating import cycles.

Gating (all global env — per-stage env / ``hf_overrides`` do NOT reach the
multimodal processor):
- ``COSYVOICE3_HF_TALKER=1``         enable HF talker mode.
- ``COSYVOICE3_HF_TALKER_PATH=<dir>`` the ``hf_cosyvoice3_llm`` checkpoint dir
  (talker weights live in a different repo than the FunAudioLLM main model that
  supplies flow/hift/onnx).
- ``COSYVOICE3_RAS=0``               force vLLM default sampling (otherwise the
  custom repetition-aware sampling is used, matching the fp32 path).
"""
from __future__ import annotations

import json
import os

# Fallbacks if the checkpoint ships no ``cosyvoice3_metadata.json`` (these match
# the convert_cosyvoice3_to_hf.py output for Fun-CosyVoice3-0.5B).
_DEFAULT_SPEECH_TOKEN_OFFSET = 151924
_DEFAULT_EOS_TOKEN_ID = 158486
_DEFAULT_PADDED_VOCAB_SIZE = 158720
_DEFAULT_BASE_SPEECH_TOKEN_SIZE = 6561

_FALSE = ("0", "false", "False", "")


def hf_talker_enabled() -> bool:
    return os.environ.get("COSYVOICE3_HF_TALKER", "0") not in _FALSE


def hf_talker_path() -> str | None:
    return os.environ.get("COSYVOICE3_HF_TALKER_PATH") or None


def ras_enabled() -> bool:
    """Repetition-aware sampling toggle (default on, matching the fp32 path)."""
    return os.environ.get("COSYVOICE3_RAS", "1") not in _FALSE


_metadata_cache: dict[str, dict] = {}


def hf_metadata() -> dict:
    """Read ``cosyvoice3_metadata.json`` from the HF talker dir (cached).

    Returns a dict with at least ``speech_token_offset``, ``eos_token_id``,
    ``padded_vocab_size`` and ``base_speech_token_size`` — falling back to the
    convert-script defaults when the file or a key is absent.
    """
    path = hf_talker_path() or ""
    if path not in _metadata_cache:
        meta: dict = {}
        meta_file = os.path.join(path, "cosyvoice3_metadata.json") if path else ""
        if meta_file and os.path.exists(meta_file):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        _metadata_cache[path] = {
            "speech_token_offset": int(meta.get("speech_token_offset", _DEFAULT_SPEECH_TOKEN_OFFSET)),
            "eos_token_id": int(meta.get("eos_token_id", _DEFAULT_EOS_TOKEN_ID)),
            "padded_vocab_size": int(meta.get("padded_vocab_size", _DEFAULT_PADDED_VOCAB_SIZE)),
            "base_speech_token_size": int(meta.get("base_speech_token_size", _DEFAULT_BASE_SPEECH_TOKEN_SIZE)),
        }
    return _metadata_cache[path]
