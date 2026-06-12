# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved
"""Model-agnostic TensorRT plumbing shared by the omni TTS accelerators.

These helpers factor out the engine-build / plan-cache / execution-context
machinery first written for CosyVoice3 (``cosyvoice3/speaker_embedding_trt.py``
and ``cosyvoice3/flow_estimator_trt.py``) so other models (e.g. Qwen3-TTS) can
reuse it without importing across model packages.

Precision notes (TensorRT >= 11): the weakly-typed ``BuilderFlag.FP16`` was
dropped, so fp16 only comes from a STRONGLY_TYPED network built from an fp16
ONNX. An fp32 ONNX is built fp32 + the TF32 matmul flag. ``EXPLICIT_BATCH`` is
implicit (no flag) and ``ITensor.dtype`` is read-only, so neither is set here.
``import tensorrt`` is kept lazy (inside functions) so environments without TRT
can still import this module.
"""

from __future__ import annotations

import hashlib
import os
import queue

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def trt_logger():
    import tensorrt as trt

    return trt.Logger(trt.Logger.WARNING)


def resolve_plan_path(
    onnx_path: str,
    *,
    prefix: str,
    cache_env_var: str,
    default_subdir: str,
) -> str:
    """Return the cached ``.plan`` path for ``onnx_path``.

    Engines are device- and TRT-version-specific, so the cache key includes the
    ONNX abspath/size/mtime, the device name, and the TRT version. ``prefix``
    should encode anything else that changes the engine (e.g. precision), so
    distinct builds never collide. Override the cache dir with ``cache_env_var``.
    """
    cache_dir = os.environ.get(cache_env_var) or os.path.join(
        os.path.expanduser("~"), ".cache", "vllm_omni", default_subdir
    )
    os.makedirs(cache_dir, exist_ok=True)
    try:
        # No-arg form targets the current device (avoids the banned
        # torch.cuda.current_device per ruff TID251).
        dev_name = torch.cuda.get_device_name()
    except Exception:
        dev_name = "unknown"
    import tensorrt as trt

    st = os.stat(onnx_path)
    key = f"{os.path.abspath(onnx_path)}|{st.st_size}|{int(st.st_mtime)}|{dev_name}|trt{trt.__version__}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"{prefix}_{digest}.plan")


def build_engine_from_onnx(
    onnx_path: str,
    plan_path: str,
    *,
    profiles: dict[str, tuple[tuple, tuple, tuple]],
    strongly_typed: bool,
    workspace_bytes: int = 1 << 32,
) -> None:
    """Parse ``onnx_path`` and serialize a TensorRT engine to ``plan_path``.

    ``profiles`` maps a dynamic input tensor name to ``(min, opt, max)`` shape
    tuples. ``strongly_typed`` builds a STRONGLY_TYPED network (the only way to
    get fp16 in TRT>=11, from an fp16 ONNX); otherwise an fp32 network with the
    TF32 matmul flag enabled. Writes atomically via ``.tmp`` + ``os.replace``.
    """
    import tensorrt as trt

    logger.info(
        "Building TensorRT engine from %s (%s) ...",
        onnx_path,
        "strongly-typed/fp16" if strongly_typed else "fp32+TF32",
    )
    tlogger = trt_logger()
    builder = trt.Builder(tlogger)
    if strongly_typed:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    else:
        # EXPLICIT_BATCH is implicit in TRT>=10; create the network with no flags.
        network = builder.create_network(0)
    parser = trt.OnnxParser(network, tlogger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise ValueError(f"Failed to parse {onnx_path}: {errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if not strongly_typed:
        # fp32 ONNX: enable the best reduced-precision matmul flag available.
        # TRT>=11 dropped the weakly-typed FP16 flag, so this is TF32 on
        # Ampere+/Hopper. A strongly-typed network fixes precision in the graph,
        # so no flag is set there.
        for _flag_name in ("FP16", "TF32"):
            _flag = getattr(trt.BuilderFlag, _flag_name, None)
            if _flag is not None:
                config.set_flag(_flag)
                break

    if profiles:
        profile = builder.create_optimization_profile()
        for name, (mn, op, mx) in profiles.items():
            profile.set_shape(name, mn, op, mx)
        config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT failed to build engine from {onnx_path}")
    tmp = plan_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(engine_bytes)
    os.replace(tmp, plan_path)
    logger.info("Wrote TensorRT engine to %s", plan_path)


def load_engine(plan_path: str):
    """Deserialize a serialized TensorRT engine from ``plan_path``."""
    import tensorrt as trt

    runtime = trt.Runtime(trt_logger())
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine {plan_path}")
    return engine


class TrtContextPool:
    """Bounded pool of TensorRT execution contexts.

    TensorRT execution contexts are not safe for concurrent ``enqueue``; a pool
    lets several callers each hold their own context up to ``concurrency``.

    Callers must run ``execute_async_v3`` on the **current** CUDA stream (the one
    the surrounding torch ops use), not a private stream — otherwise the engine
    can read its input before the torch ops that produced it have completed
    (a silent data race). The pool therefore hands out only contexts.
    """

    def __init__(self, engine, device: str | torch.device, concurrency: int = 1):
        self.engine = engine
        self._pool: queue.Queue = queue.Queue(maxsize=concurrency)
        for _ in range(concurrency):
            ctx = engine.create_execution_context()
            assert ctx is not None, "failed to create TRT execution context (out of memory?)"
            self._pool.put(ctx)

    def acquire(self):
        return self._pool.get()

    def release(self, ctx):
        self._pool.put(ctx)
