"""
M4Raw preprocessing for SGF-MRI / DCAMSR reproduction.

Everything here is numpy + h5py only (no torch), so it runs on a Kaggle CPU
session and does not depend on DCAMSR's old torch==1.7 FFT API.

What it reproduces from the DCAMSR repo (Solor-pikachu/DCAMSR):
  * pairing:   T1 rep k  <->  T2 rep k of the same scan  (dataset/M4raw_get_train_val.py)
  * HR target: the `reconstruction_rss` stored in each H5 file
  * LR input:  per-coil  k-space -> ifft2c -> centre-crop image -> fft2c ->
               centre-crop k-space to (H/s, W/s) -> ifft2c -> |.| -> RSS over coils
               (experimental/DCAMSR/DCAMSR.py::DataTransform.__call__)
  * FFTs are centred and orthonormal, same as fastmri.math.fft2c/ifft2c
    (torch.fft(..., normalized=True) == numpy norm="ortho").
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import h5py
import numpy as np

# --------------------------------------------------------------------------- #
# FFT helpers (identical semantics to DCAMSR's fastmri/math.py)
# --------------------------------------------------------------------------- #
def fft2c(x: np.ndarray) -> np.ndarray:
    """Centred orthonormal 2D FFT over the last two axes (complex input)."""
    x = np.fft.ifftshift(x, axes=(-2, -1))
    x = np.fft.fft2(x, axes=(-2, -1), norm="ortho")
    return np.fft.fftshift(x, axes=(-2, -1))


def ifft2c(x: np.ndarray) -> np.ndarray:
    """Centred orthonormal 2D inverse FFT over the last two axes."""
    x = np.fft.ifftshift(x, axes=(-2, -1))
    x = np.fft.ifft2(x, axes=(-2, -1), norm="ortho")
    return np.fft.fftshift(x, axes=(-2, -1))


def center_crop(x: np.ndarray, shape) -> np.ndarray:
    """Centre crop over the last two axes (same arithmetic as fastMRI)."""
    h, w = shape
    assert 0 < h <= x.shape[-2] and 0 < w <= x.shape[-1], (x.shape, shape)
    h0 = (x.shape[-2] - h) // 2
    w0 = (x.shape[-1] - w) // 2
    return x[..., h0:h0 + h, w0:w0 + w]


def rss(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.sqrt((np.abs(x) ** 2).sum(axis=axis))


# --------------------------------------------------------------------------- #
# DCAMSR-exact LR generation
# --------------------------------------------------------------------------- #
def dcamsr_crop_size(image_hw, target_hw):
    """DCAMSR crops to the target size, with its 'sFLAIR 203' guard."""
    crop = (target_hw[0], target_hw[1])
    if image_hw[0] < crop[1]:
        crop = (image_hw[0], image_hw[0])
    return crop


def dcamsr_lr_from_kspace(kspace_slice: np.ndarray, target_hw, scale: int) -> np.ndarray:
    """
    kspace_slice: (coils, H, W) complex  (multicoil, M4Raw)  or (H, W) (singlecoil)
    returns the LR magnitude image of shape (crop_h // scale, crop_w // scale)
    """
    img = ifft2c(kspace_slice)
    crop = dcamsr_crop_size(img.shape[-2:], target_hw)
    img = center_crop(img, crop)
    k = fft2c(img)
    k = center_crop(k, (crop[0] // scale, crop[1] // scale))
    lr = np.abs(ifft2c(k))
    if lr.ndim == 3:            # multicoil -> RSS over coils
        lr = np.sqrt((lr ** 2).sum(axis=0))
    return lr.astype(np.float32)


# --------------------------------------------------------------------------- #
# Pairing (same rule as DCAMSR's dataset/M4raw_get_train_val.py)
# --------------------------------------------------------------------------- #
NAME_RE = re.compile(r"^(?P<scan>\d{10})_(?P<seq>T1|T2|FLAIR)(?P<rep>\d{2})\.h5$")


def find_h5(root: Path) -> dict[str, Path]:
    """Map 'scanid_T101' -> path, searching recursively (zip/HF layouts differ)."""
    out = {}
    for p in Path(root).rglob("*.h5"):
        if NAME_RE.match(p.name):
            out[p.stem] = p
    return out


def build_pairs(files: dict[str, Path]):
    """Return sorted list of (T1_stem, T2_stem) pairs; also report orphans."""
    pairs, orphans = [], []
    for stem in sorted(files):
        m = NAME_RE.match(stem + ".h5")
        if m["seq"] != "T1":
            continue
        t2 = f"{m['scan']}_T2{m['rep']}"
        if t2 in files:
            pairs.append((stem, t2))
        else:
            orphans.append(stem)
    return pairs, orphans


def write_pair_csv(pairs, path: Path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(pairs)


def read_pair_csv(path: Path):
    with open(path) as f:
        return [tuple(r[:2]) for r in csv.reader(f) if r]


def compare_with_dcamsr(our_pairs, dcamsr_csv: Path, split: str):
    """Print how our split differs from the CSV DCAMSR (and SGF-MRI) used."""
    theirs = set(read_pair_csv(dcamsr_csv))
    ours = set(our_pairs)
    subj = lambda s: {a.split("_")[0] for a, _ in s}
    print(f"[{split}] ours: {len(ours)} pairs / {len(subj(ours))} subjects | "
          f"DCAMSR csv: {len(theirs)} pairs / {len(subj(theirs))} subjects")
    only_theirs = sorted(subj(theirs) - subj(ours))
    only_ours = sorted(subj(ours) - subj(theirs))
    print(f"[{split}]   subjects only in DCAMSR csv : {only_theirs}")
    print(f"[{split}]   subjects only in our files  : {only_ours}")
    return only_theirs, only_ours


# --------------------------------------------------------------------------- #
# Inspection helpers
# --------------------------------------------------------------------------- #
def inspect_file(path: Path) -> dict:
    with h5py.File(path, "r") as hf:
        k = hf["kspace"]
        info = {
            "keys": list(hf.keys()),
            "kspace_shape": k.shape,
            "kspace_dtype": str(k.dtype),
            "rss_shape": hf["reconstruction_rss"].shape if "reconstruction_rss" in hf else None,
            "attrs": {a: (v.decode() if isinstance(v, bytes) else v) for a, v in hf.attrs.items()},
        }
        mid = k.shape[0] // 2
        ks = k[mid]
        # which axis has unacquired (all-zero) lines -> phase-encode axis
        mag = np.abs(ks).sum(axis=0) if ks.ndim == 3 else np.abs(ks)
        info["zero_rows"] = int((mag.sum(axis=1) == 0).sum())
        info["zero_cols"] = int((mag.sum(axis=0) == 0).sum())
        if "reconstruction_rss" in hf:
            tgt = hf["reconstruction_rss"][mid]
            rec = center_crop(rss(ifft2c(ks), axis=0) if ks.ndim == 3 else np.abs(ifft2c(ks)), tgt.shape)
            info["rss_rel_err_mid_slice"] = float(np.linalg.norm(rec - tgt) / np.linalg.norm(tgt))
    return info


# --------------------------------------------------------------------------- #
# Cache writer: one small H5 per (T1, T2) pair
# --------------------------------------------------------------------------- #
def preprocess_pair(t1_path: Path, t2_path: Path, out_path: Path, scales=(2, 4)):
    """
    Writes:
      ref_hr  (S,H,W) float32  T1 reconstruction_rss   (reference, fully sampled)
      tar_hr  (S,H,W) float32  T2 reconstruction_rss   (target ground truth)
      ref_lr{s}, tar_lr{s}     DCAMSR-exact LR images for each scale s
    S = min(#slices T1, #slices T2) exactly like DCAMSR's SliceDataset.
    """
    with h5py.File(t1_path, "r") as h1, h5py.File(t2_path, "r") as h2:
        n = min(h1["kspace"].shape[0], h2["kspace"].shape[0])
        data = {}
        for tag, hf in (("ref", h1), ("tar", h2)):
            hr = hf["reconstruction_rss"][:n].astype(np.float32)
            data[f"{tag}_hr"] = hr
            for s in scales:
                data[f"{tag}_lr{s}"] = np.stack(
                    [dcamsr_lr_from_kspace(hf["kspace"][i], hr.shape[-2:], s) for i in range(n)]
                )
        attrs = {"ref_file": t1_path.name, "tar_file": t2_path.name, "num_slices": n,
                 "ref_max": float(h1.attrs.get("max", data["ref_hr"].max())),
                 "tar_max": float(h2.attrs.get("max", data["tar_hr"].max()))}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v, chunks=(1,) + v.shape[1:])   # 1 slice per chunk -> fast reads
        for k, v in attrs.items():
            f.attrs[k] = v
    tmp.rename(out_path)          # atomic: a half-written file never looks finished
    return n