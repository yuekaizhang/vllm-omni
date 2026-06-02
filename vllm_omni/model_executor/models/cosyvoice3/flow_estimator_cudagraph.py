# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph wrapper for the CosyVoice3 flow-decoder (CFM) DiT estimator.

Alternative to the TensorRT estimator: capture the torch DiT forward
``estimator(x, mask, mu, t, spks, cond)`` (one CFM ODE step) into CUDA graphs to
remove per-step kernel-launch overhead while staying in torch. Mirrors the
GLM-TTS DiT cudagraph wrapper (per-bucket static buffers, pad-to-bucket, replay).

The estimator has a dynamic mel-frame dim ``T`` (``x``/``mask``/``mu``/``cond``
are ``[2, *, T]``; ``t`` ``[2]``; ``spks`` ``[2, 80]``). Requests are
right-padded to the nearest captured bucket (``mask`` padding = 0 so the DiT
attention ignores it) and the output is trimmed back to ``T``. Graphs are
captured lazily per bucket on first use; ``T`` larger than the largest bucket
falls back to eager. The flow runs in fp32, where the diffusion attention uses
its SDPA fallback (cudagraph-capturable).

Enabled with ``COSYVOICE3_FLOW_CUDAGRAPH=1`` (bucket list overridable via
``COSYVOICE3_FLOW_CUDAGRAPH_SIZES``, e.g. "100,200,400,800,1600,3000").
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_BUCKETS = (100, 200, 400, 800, 1200, 1600, 2400, 3000)


def _graph_pool():
    try:
        from vllm.platforms import current_platform

        return current_platform.get_global_graph_pool()
    except Exception:
        return None


def flow_cudagraph_enabled() -> bool:
    return os.environ.get("COSYVOICE3_FLOW_CUDAGRAPH", "0") not in ("0", "false", "False", "")


def _capture_sizes() -> list[int]:
    raw = os.environ.get("COSYVOICE3_FLOW_CUDAGRAPH_SIZES")
    if raw:
        try:
            return sorted({int(s) for s in raw.split(",") if s.strip()})
        except ValueError:
            logger.warning("Bad COSYVOICE3_FLOW_CUDAGRAPH_SIZES=%r; using defaults", raw)
    return list(_DEFAULT_BUCKETS)


class CudaGraphDiTEstimator(nn.Module):
    """Drop-in for the DiT estimator that replays CUDA graphs per mel-frame bucket."""

    def __init__(self, dit: nn.Module, capture_sizes: list[int] | None = None):
        super().__init__()
        self.dit = dit
        self.capture_sizes = sorted(set(capture_sizes or _capture_sizes()))
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._sin: dict[int, dict[str, torch.Tensor]] = {}
        self._sout: dict[int, torch.Tensor] = {}
        self._disabled = False  # set if capture fails -> permanent eager fallback

    def _bucket(self, t_len: int) -> int | None:
        for s in self.capture_sizes:
            if t_len <= s:
                return s
        return None

    @torch.inference_mode()
    def _capture(self, size: int, ref: torch.Tensor) -> None:
        dev, dtype = ref.device, ref.dtype
        sin = {
            "x": torch.zeros(2, 80, size, device=dev, dtype=dtype),
            "mask": torch.zeros(2, 1, size, device=dev, dtype=dtype),
            "mu": torch.zeros(2, 80, size, device=dev, dtype=dtype),
            "t": torch.zeros(2, device=dev, dtype=dtype),
            "spks": torch.zeros(2, 80, device=dev, dtype=dtype),
            "cond": torch.zeros(2, 80, size, device=dev, dtype=dtype),
        }
        # Warm up (off the capture stream) so cuBLAS/attention workspaces are
        # allocated before capture.
        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            for _ in range(3):
                self.dit(sin["x"], sin["mask"], sin["mu"], sin["t"], sin["spks"], sin["cond"])
        torch.cuda.current_stream(dev).wait_stream(s)
        torch.cuda.synchronize(dev)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=_graph_pool()):
            out = self.dit(sin["x"], sin["mask"], sin["mu"], sin["t"], sin["spks"], sin["cond"])
        self._graphs[size] = graph
        self._sin[size] = sin
        self._sout[size] = out
        logger.info("Captured CosyVoice3 flow-estimator CUDA graph for mel frames=%d", size)

    @torch.inference_mode()
    def forward(self, x, mask, mu, t, spks, cond):
        t_len = int(x.shape[-1])
        size = None if self._disabled else self._bucket(t_len)
        if size is None:
            return self.dit(x, mask, mu, t, spks, cond)
        if size not in self._graphs:
            try:
                self._capture(size, x)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "CosyVoice3 flow-estimator CUDA graph capture failed (%s); using eager DiT", exc
                )
                self._disabled = True
                return self.dit(x, mask, mu, t, spks, cond)

        sin = self._sin[size]
        # Right-pad into the static buffers; mask padding = 0 so attention ignores it.
        sin["x"].zero_()
        sin["x"][..., :t_len].copy_(x)
        sin["mask"].zero_()
        sin["mask"][..., :t_len].copy_(mask)
        sin["mu"].zero_()
        sin["mu"][..., :t_len].copy_(mu)
        sin["t"].copy_(t.reshape(-1)[:2] if t.numel() >= 2 else t.expand(2))
        sin["spks"].copy_(spks)
        sin["cond"].zero_()
        sin["cond"][..., :t_len].copy_(cond)
        self._graphs[size].replay()
        return self._sout[size][..., :t_len].clone()
