#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved
"""Export the Qwen3-TTS 12Hz codec decoder to ONNX (dynamic batch, fixed frames).

Exports from vLLM-Omni's own ``Qwen3TTSTokenizerV2Decoder`` (no external
``qwen_tts`` dependency). The decoder weights live in the ``speech_tokenizer/``
subfolder of a Qwen3-TTS model; we load only the ``decoder.*`` tensors so the
encoder is never instantiated.

The exported graph wraps ``decoder(audio_codes.transpose(1,2)).squeeze(1)`` so
the ONNX I/O is ``audio_codes [B, F0, Q] (int64)`` -> ``audio_values [B, audio_len]
(fp32)`` (the decoder's final ``clamp(-1,1)`` is included).

Only the **batch** axis is dynamic; the frames axis is fixed to ``--frames``
(default 97 = the streaming window ``codec_chunk_frames + codec_left_context_frames``).
The decoder's causal-conv padding derives the output length from a Python ``float``
of the frame count, so the trace bakes that length in and a dynamic-frames axis
does not generalize. Instead the runtime right-pads each chunk up to F0 and
slices the output back (the causal convs make right-padding safe; this mirrors
the existing CUDA-graph decoder wrapper). A deploy that raises
``decode_chunk_frames`` must re-export + rebuild with a larger ``--frames``.
"""

# ruff: noqa: E402 — the SDPA/mask monkeypatches below must run before the
# transformers/vllm_omni model imports pull the patched callables into scope.

import argparse
import os
from pathlib import Path

import numpy as np
import torch

# Match ORT's full-FP32 matmul; PyTorch on Ampere+ uses TF32 by default.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# The HF mask builders use torch.vmap, which the TorchScript ONNX exporter cannot
# trace. Replace them with traceable builders that materialize the SAME additive
# masks as static tensors. Returning None (the naive "make it traceable" hack)
# is WRONG: it makes attention full/bidirectional, so a chunk right-padded to F0
# attends to the zero padding and the decoded audio is garbled. Because the
# frames axis is fixed (F0), these fold to constant [1,1,F0,F0] masks.
try:
    import transformers.masking_utils as _mu

    def _static_additive_mask(seq_len, dtype, device, window=None):
        idx = torch.arange(seq_len, device=device)
        q = idx[:, None]
        k = idx[None, :]
        allowed = k <= q  # causal
        if window is not None and int(window) > 0:
            allowed = allowed & (k > q - int(window))  # sliding window
        mask = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)[None, None]

    def _embeds_from_kwargs(kw):
        emb = kw.get("inputs_embeds", kw.get("input_embeds"))
        if emb is None:
            raise RuntimeError("mask builder: inputs_embeds not in kwargs")
        return emb

    def _causal_mask_builder(**kw):
        emb = _embeds_from_kwargs(kw)
        return _static_additive_mask(emb.shape[1], emb.dtype, emb.device)

    def _sliding_mask_builder(**kw):
        emb = _embeds_from_kwargs(kw)
        window = getattr(kw.get("config"), "sliding_window", None)
        return _static_additive_mask(emb.shape[1], emb.dtype, emb.device, window=window)

    if hasattr(_mu, "create_causal_mask"):
        _mu.create_causal_mask = _causal_mask_builder
    if hasattr(_mu, "create_sliding_window_causal_mask"):
        _mu.create_sliding_window_causal_mask = _sliding_mask_builder
except ImportError:
    pass

# enable_gqa=True is unsupported by the TorchScript ONNX exporter and is a no-op
# here (num_heads == num_kv_heads).
_orig_sdpa = torch.nn.functional.scaled_dot_product_attention


def _sdpa_no_gqa(*args, **kwargs):
    kwargs.pop("enable_gqa", None)
    return _orig_sdpa(*args, **kwargs)


torch.nn.functional.scaled_dot_product_attention = _sdpa_no_gqa

try:
    import onnx
except ImportError as exc:
    raise ImportError("`onnx` is required. Install with: pip install onnx onnxruntime") from exc

from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)


class CodecDecoderWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(audio_codes.transpose(1, 2)).squeeze(1)


def _resolve_model_dir(model: str) -> str:
    if os.path.isdir(model):
        return model
    return snapshot_download(model, allow_patterns=["speech_tokenizer/*"])


def build_decoder(model: str, device: torch.device) -> Qwen3TTSTokenizerV2Decoder:
    model_dir = _resolve_model_dir(model)
    config = Qwen3TTSTokenizerV2Config.from_pretrained(model_dir, subfolder="speech_tokenizer")
    # Eager attention so the additive mask path (eager_attention_forward) is used,
    # matching the static masks installed above.
    config.decoder_config._attn_implementation = "eager"
    decoder = Qwen3TTSTokenizerV2Decoder._from_config(config.decoder_config)

    weights_path = os.path.join(model_dir, "speech_tokenizer", "model.safetensors")
    state = load_file(weights_path)
    # Keep only decoder.* tensors, strip the prefix to match module names.
    dec_state = {k[len("decoder.") :]: v for k, v in state.items() if k.startswith("decoder.")}
    missing, unexpected = decoder.load_state_dict(dec_state, strict=False)
    missing = [m for m in missing if "exp_cache" not in m and "inv_exp" not in m]
    if missing:
        raise RuntimeError(f"Missing decoder weights after load: {missing[:10]} ... ({len(missing)} total)")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys ignored, e.g. {unexpected[:5]}")

    decoder.to(device=device, dtype=torch.float32).eval()
    if hasattr(decoder, "precompute_snake_caches"):
        decoder.precompute_snake_caches()
    return decoder


def check_onnx_parity(wrapper, onnx_path, audio_codes, device, atol=1e-3) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed - skipping parity check")
        return True

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    with torch.inference_mode():
        ref = wrapper(audio_codes).detach().cpu().float().numpy()
    ort_out = sess.run(None, {"audio_codes": audio_codes.cpu().numpy()})[0]
    if ref.shape != ort_out.shape:
        print(f"ONNX parity batch={audio_codes.shape[0]}: SHAPE MISMATCH torch={ref.shape} ort={ort_out.shape}")
        return False
    max_diff = float(np.abs(ref - ort_out).max())
    ok = max_diff <= atol
    print(
        f"ONNX parity batch={audio_codes.shape[0]} frames={audio_codes.shape[1]} ({sess.get_providers()[0]}): "
        f"max_abs_diff={max_diff:.6f} atol={atol} {'PASSED' if ok else 'FAILED'}"
    )
    return ok


def parse_args():
    p = argparse.ArgumentParser(description="Export Qwen3-TTS 12Hz codec decoder to ONNX (fixed frames, dynamic batch)")
    p.add_argument(
        "--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="HF id or local path of the Qwen3-TTS model"
    )
    p.add_argument("--onnx-path", default="codec.onnx")
    p.add_argument(
        "--frames",
        type=int,
        default=97,
        help="fixed frame size F0 baked into the graph. Match the streaming window "
        "codec_chunk_frames + codec_left_context_frames (=97 in qwen3_tts.yaml). "
        "Chunks larger than F0 fall back to the torch decoder at runtime.",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "fp32"],
        help="graph precision. fp16 exports a natively-fp16 graph (build strongly-typed for TRT>=11, "
        "which dropped the weakly-typed FP16 flag); fp32 keeps full precision (build with TF32).",
    )
    p.add_argument("--parity-batches", type=int, nargs="+", default=[1, 2], help="batch sizes to parity-check")
    p.add_argument(
        "--dynamic",
        action="store_true",
        help="Export a dynamic batch+frames graph via the dynamo exporter (no fixed F0, no "
        "runtime pad/slice). Requires onnxscript. Builds the TRT engine with a frames profile.",
    )
    p.add_argument(
        "--parity-frames",
        type=int,
        nargs="+",
        default=[30, 97, 300],
        help="frame counts to parity-check in --dynamic mode (verifies the frames axis generalizes).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    decoder = build_decoder(args.model, device)
    # Keep an fp32 reference for parity even when exporting fp16.
    ref_wrapper = CodecDecoderWrapper(decoder).to(device).eval()
    if args.dtype == "fp16":
        wrapper = CodecDecoderWrapper(build_decoder(args.model, device)).to(device).eval().half()
    else:
        wrapper = ref_wrapper

    nq = int(decoder.config.num_quantizers)
    codebook_size = int(decoder.config.codebook_size)

    def make_dummy(batch: int, frames: int) -> torch.Tensor:
        return torch.randint(0, codebook_size, (batch, frames, nq), dtype=torch.long, device=device)

    onnx_path = Path(args.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    atol = 0.1 if args.dtype == "fp16" else 1e-3
    all_ok = True

    if args.dynamic:
        # Dynamo exporter keeps the frames axis symbolic (the integer conv-padding
        # in the model stays a SymInt). Needs onnxscript.
        batch_dim = torch.export.Dim("batch", min=1, max=64)
        frames_dim = torch.export.Dim("frames", min=8, max=4096)
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (make_dummy(args.batch_size, args.frames),),
                str(onnx_path),
                dynamo=True,
                opset_version=args.opset,
                input_names=["audio_codes"],
                output_names=["audio_values"],
                dynamic_shapes={"audio_codes": {0: batch_dim, 1: frames_dim}},
            )
        print(f"ONNX (dynamic batch+frames) exported to {onnx_path} (dtype={args.dtype}, nq={nq})")
        onnx.checker.check_model(str(onnx_path))
        # Vary frames: the output length must scale (proves the frames axis generalizes).
        for frames in args.parity_frames:
            all_ok &= check_onnx_parity(ref_wrapper, onnx_path, make_dummy(1, frames), device, atol=atol)
    else:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (make_dummy(args.batch_size, args.frames),),
                str(onnx_path),
                dynamo=False,
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=["audio_codes"],
                output_names=["audio_values"],
                dynamic_axes={"audio_codes": {0: "batch"}, "audio_values": {0: "batch"}},
            )
        print(f"ONNX exported to {onnx_path} (dtype={args.dtype}, nq={nq}, frames=F0={args.frames})")
        onnx.checker.check_model(str(onnx_path))
        # Parity over batch sizes (frames fixed). Compare vs the fp32 reference.
        for batch in args.parity_batches:
            all_ok &= check_onnx_parity(ref_wrapper, onnx_path, make_dummy(batch, args.frames), device, atol=atol)

    if not all_ok:
        raise RuntimeError("ONNX vs PyTorch parity failed - export is broken.")


if __name__ == "__main__":
    main()
