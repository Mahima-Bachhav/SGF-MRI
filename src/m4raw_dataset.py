"""
Training-time Dataset for the preprocessed M4Raw cache (T1 reference -> T2 target).

Two tasks, matching SGF-MRI Section 3.1 / 4:
  task="sr",    scale in {2, 4}   -> LR input = DCAMSR-exact LR image (precomputed)
  task="recon", accel in {4, 8}   -> zero-filled input  Y = |F^-1( M * F(X_tar) )|
                                     with a fastMRI RandomMaskFunc column mask

Normalisation follows DCAMSR exactly: per-slice instance norm using the mean/std
of the *degraded* input, the target is normalised with the same mean/std, the
reference with its own input's statistics, then everything is clamped to [-6, 6].

Usage on Kaggle:
    import sys; sys.path.append(CACHE_DIR)        # the cache dataset ships this file
    from m4raw_dataset import M4RawPairs, evaluate_volumes
    train_ds = M4RawPairs(CACHE_DIR, "train", task="sr", scale=4)
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import h5py
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:          # lets the numpy logic be unit-tested without torch
    torch = None
    Dataset = object

from m4raw_prep import fft2c, ifft2c


# --------------------------------------------------------------------------- #
# fastMRI RandomMaskFunc, ported 1:1 (DCAMSR fastmri/data/subsample.py)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _temp_seed(rng, seed):
    if seed is None:
        yield
        return
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)


class RandomMaskFunc:
    """Fully samples num_cols*center_fraction centre columns, the rest at random."""

    def __init__(self, center_fractions, accelerations):
        assert len(center_fractions) == len(accelerations)
        self.center_fractions, self.accelerations = center_fractions, accelerations
        self.rng = np.random.RandomState()

    def __call__(self, num_cols: int, seed=None) -> np.ndarray:
        with _temp_seed(self.rng, seed):
            choice = self.rng.randint(0, len(self.accelerations))
            cf, acc = self.center_fractions[choice], self.accelerations[choice]
            num_low = int(round(num_cols * cf))
            prob = (num_cols / acc - num_low) / (num_cols - num_low)
            mask = self.rng.uniform(size=num_cols) < prob
            pad = (num_cols - num_low + 1) // 2
            mask[pad:pad + num_low] = True
        return mask.astype(np.float32)


# fastMRI convention: 4x -> 8% centre, 8x -> 4% centre. NOT stated in the SGF-MRI paper.
DEFAULT_CENTER_FRACTION = {4: 0.08, 8: 0.04}


def _seed_from_name(name: str):
    """fastMRI's deterministic per-volume seed (same mask for a volume every epoch)."""
    return tuple(map(ord, name))


def _norm(x, mean, std, eps=1e-11):
    return np.clip((x - mean) / (std + eps), -6, 6).astype(np.float32)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class M4RawPairs(Dataset):
    def __init__(self, cache_dir, split, task="sr", scale=4, accel=4,
                 center_fraction=None, deterministic_mask=None, max_pairs=None):
        """
        split: "train" or "val"  (val = the paper's 30-subject test set)
        deterministic_mask: default False for train (new random mask every time,
            like DCAMSR use_seed=False) and True for val (fixed mask per volume).
        """
        assert task in ("sr", "recon")
        self.task, self.scale, self.accel = task, scale, accel
        self.files = sorted((Path(cache_dir) / split).glob("*.h5"))
        if max_pairs:
            self.files = self.files[:max_pairs]
        if not self.files:
            raise FileNotFoundError(f"No cache files in {Path(cache_dir) / split}")
        self.examples = []
        for f in self.files:
            with h5py.File(f, "r") as hf:
                self.examples += [(f, i) for i in range(int(hf.attrs["num_slices"]))]
        if task == "recon":
            cf = center_fraction or DEFAULT_CENTER_FRACTION[accel]
            self.mask_func = RandomMaskFunc([cf], [accel])
        self.deterministic = (split != "train") if deterministic_mask is None else deterministic_mask

    def __len__(self):
        return len(self.examples)

    # ---- degradations ------------------------------------------------------
    def _zero_filled(self, img, name):
        seed = _seed_from_name(name) if self.deterministic else None
        mask = self.mask_func(img.shape[-1], seed)           # sample columns (last axis)
        k = fft2c(img.astype(np.complex64)) * mask[None, :]
        return np.abs(ifft2c(k)).astype(np.float32), mask

    def __getitem__(self, i):
        path, s = self.examples[i]
        with h5py.File(path, "r") as hf:
            tar_hr = hf["tar_hr"][s]
            ref_hr = hf["ref_hr"][s]
            if self.task == "sr":
                tar_in = hf[f"tar_lr{self.scale}"][s]
                ref_in = hf[f"ref_lr{self.scale}"][s]
                mask = np.ones(1, np.float32)
            else:
                tar_in, mask = self._zero_filled(tar_hr, path.stem)
                ref_in = ref_hr          # reference is fully sampled in reconstruction

        # ddof=1: DCAMSR uses torch's .std(), which is the unbiased estimator
        mean, std = tar_in.mean(), tar_in.std(ddof=1)
        rmean, rstd = ref_in.mean(), ref_in.std(ddof=1)
        sample = {
            "input": _norm(tar_in, mean, std),     # degraded T2   (LR or zero-filled)
            "target": _norm(tar_hr, mean, std),    # clean T2
            "ref": _norm(ref_hr, rmean, rstd),     # clean T1      (DCAMSR "targetpd")
            "ref_in": _norm(ref_in, rmean, rstd),  # degraded T1   (DCAMSR "imagepd", SR only)
            "mask": mask,
            "mean": np.float32(mean), "std": np.float32(std),
            "fname": path.stem, "slice": s,
        }
        if torch is not None:
            for k in ("input", "target", "ref", "ref_in", "mask"):
                sample[k] = torch.from_numpy(sample[k])
        return sample


# --------------------------------------------------------------------------- #
# Metrics, exactly as DCAMSR/fastMRI compute them
# --------------------------------------------------------------------------- #
def evaluate_volumes(outputs: dict, targets: dict):
    """
    outputs/targets: {fname: np.ndarray (S,H,W)} in ORIGINAL intensity units,
    i.e. de-normalised with x * std + mean.
    PSNR uses data_range = volume max; SSIM is averaged over slices with the same
    data_range; both are averaged over volumes (fastmri/evaluate.py).
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    psnrs, ssims = [], []
    for name, gt in targets.items():
        pred = outputs[name]
        dr = gt.max()
        psnrs.append(peak_signal_noise_ratio(gt, pred, data_range=dr))
        ssims.append(np.mean([structural_similarity(g, p, data_range=dr) for g, p in zip(gt, pred)]))
    return {"psnr": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
            "ssim": float(np.mean(ssims)), "ssim_std": float(np.std(ssims)),
            "n_volumes": len(psnrs)}