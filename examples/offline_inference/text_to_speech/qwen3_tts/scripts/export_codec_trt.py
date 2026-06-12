#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved
"""Build a TensorRT engine from the Qwen3-TTS codec decoder ONNX.

Builds with the **Python TensorRT API** (``vllm_omni...common.trt_utils``), not
``trtexec`` — the engine must match the runtime's TRT version, and a pip-wheel
TRT install ships no ``trtexec`` (and a system ``trtexec`` is often a different
major version, whose serialized engine the runtime then refuses to load).

Precision (TRT>=11 dropped the weakly-typed FP16 builder flag):
  * ``--fp16`` (default) expects a **natively-fp16 ONNX** (export with
    ``export_codec_onnx.py --dtype fp16``) and builds a STRONGLY_TYPED network.
  * ``--fp32`` builds an fp32 network with the TF32 matmul flag.

Optional *fusion barrier*: ``Add(x, runtime_zero)`` nodes around the
post-transformer permute, to defeat a TRT-10.x fused-tactic bug that silently
produced wrong audio at dynamic batch > 1. Off by default (not observed on
TRT>=11); enable with ``--fusion-barrier`` if a build mis-fuses at batch > 1.
The target node is located by name (``--barrier-tensor``) with a structural
fallback (the ``[0,2,1]`` Transpose feeding the first upsample conv).

The frames axis is fixed (F0, baked by the export trace); the engine is rebuilt
if a deploy raises ``decode_chunk_frames`` beyond ``--frames-profile``.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference

from vllm_omni.model_executor.models.common.trt_utils import build_engine_from_onnx


def _make_runtime_zero(np_type, base_name):
    """0-D ``zero = seed - seed`` (seed=1). Hidden from the constant-folder."""
    seed = gs.Constant(name=f"{base_name}_seed", values=np.array(1, dtype=np_type))
    zero = gs.Variable(name=f"{base_name}_zero", dtype=np_type, shape=())
    sub = gs.Node(op="Sub", inputs=[seed, seed], outputs=[zero], name=f"{base_name}_zero_sub")
    return sub, zero


def _make_add_zero_barrier(tensor, zero_var, name):
    new_tensor = gs.Variable(name=f"{tensor.name}__{name}", dtype=tensor.dtype, shape=tensor.shape)
    add = gs.Node(op="Add", inputs=[tensor, zero_var], outputs=[new_tensor], name=name)
    return add, new_tensor


def _find_target_transpose(graph, named_tensor):
    """Locate the post-transformer permute node to wrap.

    Prefer the explicitly named output tensor; otherwise pick, among Transpose
    nodes with perm==[0,2,1], the one whose output is consumed by a Conv /
    ConvTranspose (the entry to the upsample/decoder stack).
    """
    by_name = [n for n in graph.nodes for o in n.outputs if o.name == named_tensor]
    if by_name:
        return by_name[0]

    consumers: dict[str, list] = {}
    for node in graph.nodes:
        for inp in node.inputs:
            consumers.setdefault(inp.name, []).append(node)

    candidates = []
    for node in graph.nodes:
        if node.op != "Transpose":
            continue
        perm = node.attrs.get("perm")
        if perm is not None and list(perm) != [0, 2, 1]:
            continue
        out_name = node.outputs[0].name
        downstream = consumers.get(out_name, [])
        if any(c.op in ("Conv", "ConvTranspose") for c in downstream):
            candidates.append(node)

    if not candidates:
        all_tp = [n.name for n in graph.nodes if n.op == "Transpose"]
        raise RuntimeError(
            f"Could not locate the post-transformer Transpose (named {named_tensor!r} not found, "
            f"no [0,2,1] Transpose feeds a Conv). Transpose nodes present: {all_tp}"
        )
    if len(candidates) > 1:
        print(
            f"  [warn] {len(candidates)} candidate Transpose nodes feed a Conv: {[c.name for c in candidates]}; using the first"
        )
    return candidates[0]


def apply_trt_fusion_barrier(onnx_path, target_tensor_name):
    """Wrap the post-transformer permute with ``Add(x, runtime_zero)`` barriers."""
    model = onnx.load(str(onnx_path))
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] onnx shape_inference failed ({exc})")
    graph = gs.import_onnx(model)

    tp_node = _find_target_transpose(graph, target_tensor_name)
    if tp_node.op != "Transpose":
        print(f"  [warn] target node {tp_node.name!r} is {tp_node.op!r}, expected Transpose; proceeding anyway")

    in_tensor = tp_node.inputs[0]
    out_tensor = tp_node.outputs[0]
    if in_tensor.dtype is None:
        raise RuntimeError(f"cannot insert barrier on {in_tensor.name!r}: dtype unknown")
    np_type = np.dtype(in_tensor.dtype).type
    safe_name = tp_node.name.lstrip("/").replace("/", "_") or "Transpose"

    zero_sub, zero_var = _make_runtime_zero(np_type, base_name=f"FusionBarrier_{safe_name}")
    pre_add, pre_out = _make_add_zero_barrier(in_tensor, zero_var, name=f"FusionBarrier_pre_{safe_name}")
    tp_node.inputs[0] = pre_out
    post_add, post_out = _make_add_zero_barrier(out_tensor, zero_var, name=f"FusionBarrier_post_{safe_name}")
    for node in graph.nodes:
        if node is post_add:
            continue
        for i, inp in enumerate(node.inputs):
            if inp is out_tensor:
                node.inputs[i] = post_out
    for i, outp in enumerate(graph.outputs):
        if outp is out_tensor:
            graph.outputs[i] = post_out

    graph.nodes.extend([zero_sub, pre_add, post_add])
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), str(onnx_path))
    print(f"  wrapped {tp_node.name!r} with Add(x, runtime_zero) barriers")


def _infer_num_quantizers(onnx_path):
    model = onnx.load(str(onnx_path))
    for inp in model.graph.input:
        if inp.name != "audio_codes":
            continue
        dims = inp.type.tensor_type.shape.dim
        if len(dims) >= 3 and dims[2].dim_value > 0:
            return int(dims[2].dim_value)
    raise RuntimeError(f"could not infer num_quantizers from {onnx_path} (audio_codes dim 2 not static)")


def parse_args():
    p = argparse.ArgumentParser(description="Build a TensorRT engine from the Qwen3-TTS codec ONNX")
    p.add_argument("--onnx-path", required=True)
    p.add_argument("--trt-path", required=True)
    p.add_argument("--batch-profile", nargs=3, type=int, default=[1, 8, 32], metavar=("MIN", "OPT", "MAX"))
    # The ONNX frames dim is static (baked by the trace), so min=opt=max=F0.
    # Default 97 = streaming window (codec_chunk_frames + codec_left_context_frames).
    p.add_argument("--frames-profile", nargs=3, type=int, default=[97, 97, 97], metavar=("MIN", "OPT", "MAX"))
    p.add_argument(
        "--fp32", action="store_true", help="Build a pure FP32 (TF32) engine (default: strongly-typed FP16)."
    )
    p.add_argument(
        "--fusion-barrier",
        action="store_true",
        help="Insert the post-transformer Add(x, runtime_zero) barrier (only needed if a build mis-fuses at batch > 1).",
    )
    p.add_argument(
        "--barrier-tensor",
        default="/decoder/Transpose_19_output_0",
        help="Output tensor name of the post-transformer permute to wrap (structural fallback if absent).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx_path)
    trt_path = Path(args.trt_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    if args.fusion_barrier:
        apply_trt_fusion_barrier(onnx_path, args.barrier_tensor)

    nq = _infer_num_quantizers(onnx_path)
    print(f"num_quantizers={nq} (from {onnx_path})")

    bp, fp = tuple(args.batch_profile), tuple(args.frames_profile)
    profiles = {
        "audio_codes": (
            (bp[0], fp[0], nq),
            (bp[1], fp[1], nq),
            (bp[2], fp[2], nq),
        )
    }
    trt_path.parent.mkdir(parents=True, exist_ok=True)
    # fp16 requires a strongly-typed network built from a natively-fp16 ONNX.
    build_engine_from_onnx(str(onnx_path), str(trt_path), profiles=profiles, strongly_typed=not args.fp32)
    print(f"TensorRT engine saved to {trt_path}")


if __name__ == "__main__":
    main()
