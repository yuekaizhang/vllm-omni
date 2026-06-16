# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved
"""TensorRT accelerator for the Qwen3-TTS 12Hz codec decoder.

The codec decoder (``Qwen3TTSTokenizerV2Decoder.forward``) turns audio codec
tokens ``[B, Q, F]`` into a waveform ``[B, 1, wav_len]`` and dominates Code2Wav
(Stage 1) latency. This module wraps a TensorRT engine for that per-chunk
forward so ``chunked_decode`` / ``batched_chunked_decode`` (which both call
``self(codes_chunk)``) run on TRT with no other change.

Engine I/O contract (matches the ONNX export wrapper
``decoder(audio_codes.transpose(1,2)).squeeze(1)``): input ``audio_codes``
``[B, F, Q]`` (int32/int64), output ``audio_values`` ``[B, audio_len]`` (the
graph already includes the final ``clamp(-1, 1)``). ``__call__`` receives
``[B, Q, F]`` from ``forward``, transposes to ``[B, F, Q]``, runs the engine,
and ``unsqueeze(1)``s the output back to ``[B, 1, wav_len]``.

The ``.plan`` is built lazily from the ONNX (or a prebuilt plan is deserialized
directly) and cached per device/TRT-version/precision. Set
``QWEN3_TTS_TRT_CACHE`` to override the cache dir.
"""

from __future__ import annotations

import os

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.common.trt_utils import (
    TrtContextPool,
    build_engine_from_onnx,
    load_engine,
    resolve_plan_path,
)

logger = init_logger(__name__)

_CACHE_ENV_VAR = "QWEN3_TTS_TRT_CACHE"
_CACHE_SUBDIR = "qwen3_tts_trt"


def _codec_trt_enabled() -> bool:
    """``QWEN3_TTS_CODEC_TRT`` env toggle (default on, like ``COSYVOICE3_TRT``)."""
    return os.environ.get("QWEN3_TTS_CODEC_TRT", "1") not in ("0", "false", "False", "")


def _trt_concurrency() -> int:
    try:
        return max(1, int(os.environ.get("QWEN3_TTS_CODEC_TRT_CONCURRENT", "2")))
    except ValueError:
        return 2


def _is_fp16_onnx(onnx_path: str) -> bool:
    """fp16 ONNXes are exported strongly-typed and named ``*fp16*``/``*autocast*``."""
    name = os.path.basename(onnx_path).lower()
    return "fp16" in name or "autocast" in name


def _trt_dtype_to_torch(trt_dtype) -> torch.dtype:
    import tensorrt as trt

    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int64: torch.int64,
        trt.int32: torch.int32,
    }
    bf16 = getattr(trt, "bfloat16", None)
    if bf16 is not None:
        mapping[bf16] = torch.bfloat16
    if trt_dtype not in mapping:
        raise ValueError(f"Unsupported TRT tensor dtype {trt_dtype!r}")
    return mapping[trt_dtype]


class CodecDecoderTRT:
    """TensorRT-backed drop-in for ``Qwen3TTSTokenizerV2Decoder.forward``."""

    def __init__(
        self,
        *,
        device: str | torch.device,
        num_quantizers: int,
        total_upsample: int,
        torch_forward,
        onnx_path: str | None = None,
        plan_path: str | None = None,
    ):
        import tensorrt as trt

        self.device = torch.device(device)
        self.num_quantizers = int(num_quantizers)
        self.total_upsample = int(total_upsample)
        # Stashed torch forward for the stream-capture fallback (execute_async_v3
        # is illegal while an outer CUDA-graph stream capture is active).
        self._torch_forward = torch_forward

        if plan_path:
            engine = load_engine(plan_path)
            source = plan_path
        elif onnx_path:
            strongly_typed = _is_fp16_onnx(onnx_path)
            prefix = f"qwen3_codec_{'fp16' if strongly_typed else 'fp32'}"
            cached_plan = resolve_plan_path(
                onnx_path,
                prefix=prefix,
                cache_env_var=_CACHE_ENV_VAR,
                default_subdir=_CACHE_SUBDIR,
            )
            if not os.path.exists(cached_plan) or os.path.getsize(cached_plan) == 0:
                # The ONNX frames dim is static (F0); only the batch axis is
                # dynamic. min=opt=max on frames matches the static dim.
                f0 = _max_profile_frames()
                profiles = {
                    "audio_codes": (
                        (1, f0, self.num_quantizers),
                        (8, f0, self.num_quantizers),
                        (32, f0, self.num_quantizers),
                    )
                }
                build_engine_from_onnx(
                    onnx_path,
                    cached_plan,
                    profiles=profiles,
                    strongly_typed=strongly_typed,
                )
            engine = load_engine(cached_plan)
            source = cached_plan
        else:
            raise ValueError("CodecDecoderTRT requires onnx_path or plan_path")

        self.engine = engine
        # Resolve I/O tensor names/dtypes by mode (robust to ordering).
        self.input_name = None
        self.output_name = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        if self.input_name is None or self.output_name is None:
            raise RuntimeError(f"Codec TRT engine {source} missing input/output tensor")
        self.input_dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(self.input_name))
        self.output_dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(self.output_name))

        # Two engine kinds:
        #  * dynamic-frames (dynamo export): input [batch, -1, Q]; run at native
        #    frame count, no padding. `frame_size` = the profile's max frames
        #    (chunks beyond it fall back to torch).
        #  * fixed-frames (legacy F0): input [batch, F0, Q]; right-pad each chunk
        #    up to F0 and slice the output back (causal convs make this safe).
        in_shape = engine.get_tensor_shape(self.input_name)
        fdim = int(in_shape[1])
        self.dynamic_frames = fdim <= 0
        if self.dynamic_frames:
            try:
                _mn, _opt, _mx = engine.get_tensor_profile_shape(self.input_name, 0)
                self.frame_size = int(_mx[1])
                self.min_frames = int(_mn[1])
            except Exception:
                self.frame_size = _max_profile_frames()
                self.min_frames = 1
        else:
            self.frame_size = fdim
            self.min_frames = 1  # any f in [1, F0] is right-padded up to F0

        self.pool = TrtContextPool(engine, device=self.device, concurrency=_trt_concurrency())
        logger.info(
            "Loaded Qwen3-TTS codec TensorRT engine (%s): in=%s[%s] out=%s[%s] frames=%s(max=%d) concurrency=%d",
            source,
            self.input_name,
            self.input_dtype,
            self.output_name,
            self.output_dtype,
            "dynamic" if self.dynamic_frames else "fixed",
            self.frame_size,
            _trt_concurrency(),
        )

    @torch.inference_mode()
    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        """``codes``: ``[B, Q, F]``. Returns waveform ``[B, 1, wav_len]`` (fp32)."""
        # Inner TRT enqueue is illegal during an outer stream capture (vLLM's
        # Stage-1 FULL-cudagraph warmup). Fall back to the torch decoder there.
        if torch.cuda.is_current_stream_capturing():
            return self._torch_forward(codes)

        if codes.shape[1] != self.num_quantizers:
            raise ValueError(f"Expected {self.num_quantizers} layers of codes, got {codes.shape[1]}")

        orig_codes = codes
        b = int(codes.shape[0])
        f = int(codes.shape[2])
        # Chunks outside the engine's supported frame range fall back to the torch
        # decoder: above the profile max (rare; size tuned to the streaming window),
        # or — for a dynamic engine — below the profile min (the tiny initial/ramp
        # streaming chunks). A profile violation otherwise corrupts execution.
        if f > self.frame_size or f < self.min_frames:
            return self._torch_forward(codes)

        if self.dynamic_frames:
            run_frames = f  # run at native length, no padding
        else:
            # Right-pad up to F0 (causal convs => the first f*upsample output
            # samples are unaffected by the trailing zero frames); slice back below.
            run_frames = self.frame_size
            if f < self.frame_size:
                padded = codes.new_zeros((b, self.num_quantizers, self.frame_size))
                padded[:, :, :f] = codes
                codes = padded
        # [B, Q, run_frames] -> [B, run_frames, Q] to match the exported ONNX input.
        x = codes.transpose(1, 2).contiguous().to(self.device, dtype=self.input_dtype)
        out = torch.empty((b, run_frames * self.total_upsample), device=self.device, dtype=self.output_dtype)

        # Execute on the CURRENT stream so the engine is correctly ordered after
        # the torch ops that produced `x` and before the caller reads `out` — all
        # stream-ordered, no host sync needed (a private stream would race with
        # those ops; an explicit synchronize would stall the pipeline). execute_
        # async_v3 captures the I/O addresses at enqueue, so the context can be
        # released and reused for the next (host-sequential) call immediately.
        stream = torch.cuda.current_stream(self.device)
        ctx = self.pool.acquire()
        try:
            # A failed set_input_shape (e.g. shape outside the profile) must NOT be
            # followed by execute — that reads stale bindings and corrupts the
            # device (illegal memory access). Fall back to torch instead.
            if not ctx.set_input_shape(self.input_name, tuple(x.shape)):
                return self._torch_forward(orig_codes)
            ctx.set_tensor_address(self.input_name, x.data_ptr())
            ctx.set_tensor_address(self.output_name, out.data_ptr())
            if not ctx.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("Qwen3-TTS codec TensorRT execute_async_v3 failed")
        finally:
            self.pool.release(ctx)

        wav = out[:, : f * self.total_upsample]
        wav = wav if wav.dtype == torch.float32 else wav.to(torch.float32)
        return wav.unsqueeze(1).contiguous()


def _max_profile_frames() -> int:
    """Max frames the engine's optimization profile must cover.

    Defaults to the streaming window ``codec_chunk_frames +
    codec_left_context_frames`` (= 97 in qwen3_tts.yaml). Override with
    ``QWEN3_TTS_CODEC_MAX_FRAMES`` to match a deploy's window (the engine is
    rebuilt to match); chunks larger than F0 fall back to the torch decoder.
    Only used for the build-from-ONNX path — a prebuilt plan carries its own F0.
    """
    try:
        return max(1, int(os.environ.get("QWEN3_TTS_CODEC_MAX_FRAMES", "97")))
    except ValueError:
        return 97


# Process-wide cache: the Code2Wav stage worker is a single long-lived process;
# caching by (engine path, device) avoids re-deserializing the engine and
# re-allocating contexts. Keyed by the ONNX/plan source path.
_CODEC_TRT_CACHE: dict[tuple[str, str], CodecDecoderTRT] = {}


def get_codec_decoder_trt(
    *,
    device: str | torch.device,
    num_quantizers: int,
    total_upsample: int,
    torch_forward,
    onnx_path: str | None = None,
    plan_path: str | None = None,
) -> CodecDecoderTRT:
    """Return a process-wide cached :class:`CodecDecoderTRT`."""
    source = plan_path or onnx_path or ""
    key = (os.path.abspath(source), str(device))
    inst = _CODEC_TRT_CACHE.get(key)
    if inst is None:
        inst = CodecDecoderTRT(
            device=device,
            num_quantizers=num_quantizers,
            total_upsample=total_upsample,
            torch_forward=torch_forward,
            onnx_path=onnx_path,
            plan_path=plan_path,
        )
        _CODEC_TRT_CACHE[key] = inst
    return inst
