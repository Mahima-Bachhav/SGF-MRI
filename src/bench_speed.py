"""
Speed / memory benchmark: normal PyTorch ("eager") vs torch.compile, same maths.

Questions it answers
  1. Is SGF-MRI slower than DCAMSR only because of our plain PyTorch implementation?
     (MCNS = many small element-wise steps; the compiler fuses them into few GPU kernels.)
  2. Can compilation make future TRAINING runs cheaper (reconstruction, ablations)?
  3. Does the compiled model give the same output?

Inference: batch 1, float32 (as the paper's efficiency figure), timed with CUDA events.
Training : batch 4, mixed precision + Adam + L1 (exactly our training step), timed per iteration.
"""
from __future__ import annotations

import gc
import time

import torch
import torch.nn.functional as F


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _reset_mem(dev):
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def _peak_mb(dev):
    return torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else float("nan")


def inputs(scale, size, batch, dev):
    h = size // scale
    return (torch.randn(batch, 1, h, h, device=dev), torch.randn(batch, 1, size, size, device=dev),
            torch.randn(batch, 1, h, h, device=dev))


@torch.no_grad()
def bench_inference(model, scale, size, dev, repeats=50, warmup=10):
    """Median milliseconds per slice and peak memory (MB) for batch-1 float32 inference."""
    model.eval()
    lr, ref, ref_down = inputs(scale, size, 1, dev)
    t0 = time.time()
    for _ in range(warmup):                        # includes compilation for a compiled model
        model(lr, ref, ref_down)
    _sync(dev)
    warm_s = time.time() - t0
    _reset_mem(dev)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        model(lr, ref, ref_down)
        _sync(dev)
        times.append(time.perf_counter() - t)
    times.sort()
    return {"ms": 1000 * times[len(times) // 2], "peak_MB": _peak_mb(dev), "warmup_s": warm_s}


def bench_train(model, scale, size, dev, batch=4, iters=30, warmup=8):
    """Seconds per training iteration (batch 4, AMP, Adam, L1) and peak memory (MB)."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    lr, ref, ref_down = inputs(scale, size, batch, dev)
    tgt = torch.randn(batch, 1, size, size, device=dev)

    def step():
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(lr, ref, ref_down)
        loss = F.l1_loss(out.float(), tgt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

    t0 = time.time()
    for _ in range(warmup):
        step()
    _sync(dev)
    warm_s = time.time() - t0
    _reset_mem(dev)
    t = time.perf_counter()
    for _ in range(iters):
        step()
    _sync(dev)
    return {"s_per_it": (time.perf_counter() - t) / iters, "peak_MB": _peak_mb(dev), "warmup_s": warm_s}


@torch.no_grad()
def same_output(eager, compiled, scale, size, dev):
    """Largest absolute difference between eager and compiled outputs, relative to the output range."""
    eager.eval(); compiled.eval()
    lr, ref, ref_down = inputs(scale, size, 1, dev)
    a, b = eager(lr, ref, ref_down), compiled(lr, ref, ref_down)
    return float((a - b).abs().max() / (a.max() - a.min()).clamp_min(1e-12))


def try_compile(model, mode=None):
    """torch.compile with a readable failure instead of a crash."""
    try:
        return torch.compile(model, mode=mode), None
    except Exception as e:                          # compile itself is lazy; errors usually surface on first call
        return None, f"{type(e).__name__}: {e}"


def run(build, names, sizes, scale, dev, do_train=True, train_size=256, modes=(None,), log=print):
    """
    build(name, size) -> fresh nn.Module on `dev` (same weights reused for eager and compiled).
    Returns a list of result rows (dicts).
    """
    rows = []
    for name in names:
        for size in sizes:
            model = build(name, size)
            r = bench_inference(model, scale, size, dev)
            rows.append({"part": "inference", "model": name, "image": f"{size}x{size}", "version": "normal",
                         "ms per slice": r["ms"], "peak MB": r["peak_MB"]})
            log(f"{name} {size}px normal: {r['ms']:.1f} ms, {r['peak_MB']:.0f} MB")
            for mode in modes:
                label = f"compiled ({mode or 'default'})"
                comp, err = try_compile(model, mode)
                try:
                    if err:
                        raise RuntimeError(err)
                    r = bench_inference(comp, scale, size, dev)
                    diff = same_output(model, comp, scale, size, dev)
                    rows.append({"part": "inference", "model": name, "image": f"{size}x{size}", "version": label,
                                 "ms per slice": r["ms"], "peak MB": r["peak_MB"], "compile s": r["warmup_s"],
                                 "max output difference": diff})
                    log(f"{name} {size}px {label}: {r['ms']:.1f} ms, {r['peak_MB']:.0f} MB "
                        f"(compile+warm-up {r['warmup_s']:.0f} s, output diff {diff:.1e})")
                except Exception as e:
                    rows.append({"part": "inference", "model": name, "image": f"{size}x{size}", "version": label,
                                 "error": f"{type(e).__name__}: {str(e)[:200]}"})
                    log(f"{name} {size}px {label}: FAILED {type(e).__name__}: {str(e)[:200]}")
                del comp
                torch._dynamo.reset()
            del model; _reset_mem(dev)

        if do_train:
            for label in ("normal", "compiled (default)"):
                model = build(name, train_size)
                try:
                    m = torch.compile(model) if label != "normal" else model
                    r = bench_train(m, scale, train_size, dev)
                    rows.append({"part": "training", "model": name, "image": f"{train_size}x{train_size}",
                                 "version": label, "s per iteration": r["s_per_it"], "peak MB": r["peak_MB"],
                                 "compile s": r["warmup_s"] if label != "normal" else None})
                    log(f"{name} training {label}: {r['s_per_it']:.3f} s/it, {r['peak_MB']:.0f} MB")
                except Exception as e:
                    rows.append({"part": "training", "model": name, "image": f"{train_size}x{train_size}",
                                 "version": label, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                    log(f"{name} training {label}: FAILED {type(e).__name__}: {str(e)[:200]}")
                del model
                torch._dynamo.reset(); _reset_mem(dev)
    return rows