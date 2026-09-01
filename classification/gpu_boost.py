#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 gpu_boost.py  --  shared GPU/CPU performance setup for the CNT revision runs
================================================================================

Import this first in every script:

    import gpu_boost
    dev = gpu_boost.setup()            # TF32, cudnn autotuner, thread counts
    gpu_boost.report()

WHAT IT ACTUALLY CHANGES, AND WHY
---------------------------------
  TF32 matmul + cudnn        Blackwell runs fp32 matmuls through tensor cores at
                             TF32 precision when allowed. Off by default in
                             recent torch. Roughly 1.3-2x on conv/matmul work,
                             with no accuracy impact that matters at our scale.
  cudnn.benchmark            Autotunes conv algorithms for fixed input shapes.
                             Every workload here has fixed shapes, so this is
                             free after the first few batches.
  channels_last              Convolutional nets hit tensor cores far more
                             effectively in NHWC. Applies to ResNet, ConvNeXt,
                             VGG; a no-op for plain ViTs.
  thread counts              torch defaults to one thread per core, which
                             oversubscribes when DataLoader workers are also
                             running. Set explicitly.
  prefetching loader         The real bottleneck in feature extraction is not
                             the GPU, it is single-threaded TIFF decode blocking
                             the main loop. PIL releases the GIL while decoding,
                             so a thread pool overlaps decode with compute.
================================================================================
"""
from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_STATE = {}


def setup(deterministic: bool = False, threads: int | None = None):
    """Configure torch for throughput. Returns the resolved device string."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cudnn.benchmark = not deterministic
        if deterministic:
            torch.backends.cudnn.deterministic = True

    n = threads or max(1, (os.cpu_count() or 8))
    # leave headroom for DataLoader workers rather than claiming every core
    torch.set_num_threads(max(1, n - 2))
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, n - 2)))

    _STATE["device"] = dev
    _STATE["cores"] = n
    return dev


def workers(reserve: int = 2) -> int:
    """Sensible DataLoader worker count for this machine."""
    n = _STATE.get("cores") or (os.cpu_count() or 8)
    return max(2, min(12, n - reserve))


def to_channels_last(model):
    """NHWC for convolutional backbones. Harmless for ViTs."""
    import torch
    try:
        return model.to(memory_format=torch.channels_last)
    except Exception:
        return model


def report():
    import torch
    print("\n" + "=" * 74)
    print("COMPUTE")
    print("=" * 74)
    print(f"  torch {torch.__version__}  cuda build {torch.version.cuda}  "
          f"cores {_STATE.get('cores')}  dataloader workers {workers()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        sm = f"sm_{cap[0]}{cap[1]}"
        free, total = torch.cuda.mem_get_info()
        print(f"  {torch.cuda.get_device_name(0)}  {sm}  "
              f"{total/1e9:.1f} GB ({free/1e9:.1f} free)")
        arch = torch.cuda.get_arch_list()
        if sm not in arch:
            raise SystemExit(
                f"  FATAL: this torch has no {sm} code (has {' '.join(arch)}).\n"
                f"  pip install torch torchvision --index-url "
                f"https://download.pytorch.org/whl/cu128")
        print(f"  TF32 matmul {torch.backends.cuda.matmul.allow_tf32}   "
              f"cudnn.benchmark {torch.backends.cudnn.benchmark}   {sm} verified")
    else:
        print("  CPU only - every run below will be several times slower.")


# ---------------------------------------------------------------- prefetching
def prefetch(items, batch_size, load_fn, n_workers=None, depth=3):
    """Yield (batch_items, loaded) with decoding overlapped onto a thread pool.

    `load_fn(batch_items) -> anything`. Decode of batch k+1..k+depth proceeds
    while the GPU works on batch k, which is the single biggest win in the
    feature-extraction stages: TIFF decode was blocking the main loop and the
    GPU sat idle between batches.
    """
    nw = n_workers or workers()
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def producer():
        with ThreadPoolExecutor(max_workers=nw) as ex:
            futs = {}
            nxt = 0
            for i in range(min(depth + nw, len(batches))):
                futs[i] = ex.submit(load_fn, batches[i]); nxt = i + 1
            done = 0
            while done < len(batches) and not stop.is_set():
                r = futs.pop(done).result()
                q.put((batches[done], r))
                done += 1
                if nxt < len(batches):
                    futs[nxt] = ex.submit(load_fn, batches[nxt]); nxt += 1
        q.put(None)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            yield item
    finally:
        stop.set()


def xgb_params(base: dict, device: str, verbose: bool = True) -> dict:
    """Return xgboost params with GPU actually verified, not merely requested.

    Setting device='cuda' does not guarantee GPU execution: a CPU-only wheel
    accepts the parameter and silently trains on CPU. This trains three rounds
    and reports what really happened.
    """
    p = dict(base)
    if device != "cuda":
        p.pop("device", None)
        return p
    try:
        import xgboost as xgb
        import numpy as _np
        X = _np.random.rand(64, 8).astype("float32")
        y = (X[:, 0] > 0.5).astype(int)
        d = xgb.QuantileDMatrix(X, label=y, max_bin=16)
        xgb.train({"objective": "binary:logistic", "tree_method": "hist",
                   "device": "cuda", "max_bin": 16}, d, num_boost_round=3)
        p["device"] = "cuda"
        if verbose:
            print("  xgboost: GPU verified")
    except Exception as e:
        p.pop("device", None)
        if verbose:
            print(f"  xgboost: GPU unavailable ({str(e)[:70]}) -> CPU")
    return p
