"""
SGF-MRI (Zheng et al., Pattern Recognition 180 (2026) 114188), implemented from the paper.

The authors' code is not public (repository returns 404), so this follows Sections 3.2-3.3,
Eqs. (1)-(6), Figs. 1-2 and Section 4.1. Every choice the paper leaves open is marked
DECISION below and listed in the implementation spec.

Structure (paper Fig. 1a, "We adopted the backbone framework from DCAMSR"):
    lrsr = upsample(LR target)                     (identity for reconstruction)
    E_tar = Encoder(lrsr),  E_ref = Encoder(ref)   3 levels: 256, 128, 64 px, 64 channels
    F_i   = MCSD_i(E_tar_i, E_ref_i)               one MCSD block per level (replaces DCAMSR's attention)
    out   = Decoder(E_tar, F) + lrsr               DCAMSR decoder incl. its fusion modules (SAM)

MCSD (Fig. 1b, Fig. 2):
    SAN   saliency s = sigmoid(conv1x1(LeakyReLU(conv3x3(E))))       per contrast, n groups      (Sec 3.3.1)
    MCNS  S_ab(j,k)_c = exp(-||f^a_jc - f^b_kc||^2 / (sigma_j^2 + eps))  Eqs. (1)-(2)
          for (a,b) in (tar,tar), (tar,ref), (ref,tar), (ref,ref); k over 8 neighbours
          F_distilled = [S_tt*s_tar, S_tr*s_ref, S_rt*s_tar, S_rr*s_ref]  Eq. (3), 4*m*n channels
          F_squeezed  = W_squeeze F_distilled  (1x1 conv -> d' = 64)      Eq. (4)
    SFA   E = conv(concat(E_tar, E_ref))  ("modality fusion")
          Q = Wq F_squeezed, K = Wk E, V = Wv E                           Eq. (5)
          F_fusion = E + Wo softmax(Q K^T / sqrt(d_k)) V                  Eq. (6), local windows
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models_dcamsr import Decoder, Encoder   # the verified DCAMSR backbone (bit-identical port)

# "the two closest neighbor tokens in the up, bottom, left, and right directions" -> m = 8
NEIGHBOURS = [(-1, 0), (-2, 0), (1, 0), (2, 0), (0, -1), (0, -2), (0, 1), (0, 2)]   # (dy, dx)
PAD = 2


def neighbour(x, dy, dx):
    """x at position (y+dy, x+dx) for every (y, x). DECISION: reflect padding at the image border."""
    H, W = x.shape[-2:]
    xp = F.pad(x, (PAD, PAD, PAD, PAD), mode="reflect")
    return xp[..., PAD + dy: PAD + dy + H, PAD + dx: PAD + dx + W]


class LayerNorm2d(nn.Module):
    """LayerNorm over channels at every pixel (as nn.LayerNorm on tokens)."""

    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# --------------------------------------------------------------------------- #
# SAN: Saliency Adaptive Neighborhood (Sec. 3.3.1)
# --------------------------------------------------------------------------- #
class SAN(nn.Module):
    """'a 3x3 convolutional layer followed by a leaky ReLU activation layer and then an 1x1
    convolutional layer', one saliency score per channel group per token.
    DECISION: hidden width 32; sigmoid output so scores are in (0, 1) (paper: a 'magnitude')."""

    def __init__(self, d=64, groups=8, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(d, hidden, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(hidden, groups, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x))                       # (B, n, H, W)


# --------------------------------------------------------------------------- #
# MCNS: Multi-Contrast Neighborhood Similarity (Sec. 3.3.2, Eqs. 1-3)
# --------------------------------------------------------------------------- #
def neighbourhood_similarity(fa, fb, groups, eps=1e-6, sal_b=None):
    """
    S_ab(j,k)_c for every token j of contrast a and its 8 neighbours k in contrast b.
      fa, fb : (B, C, H, W) features, C split into `groups` contiguous channel groups
      sal_b  : (B, groups, H, W) saliency of the NEIGHBOUR contrast b, taken at each neighbour k
               (Fig. 2: the saliency map over the neighbourhood multiplies the similarity map)
    sigma_j^2: 'calculated over the local spatial neighborhood (all k) of the current token j'.
      DECISION: the mean of the 8 squared distances, per group (as in MIND / Mok et al. [15]).
    Returns (B, 8*groups, H, W), ordered [neighbour][group]. Runs in float32 for stability.
    """
    B, C, H, W = fa.shape
    g = groups
    fa = fa.float(); fb = fb.float()
    d2 = torch.stack([((fa - neighbour(fb, dy, dx)) ** 2).view(B, g, C // g, H, W).sum(2)
                      for dy, dx in NEIGHBOURS], 1)            # (B, 8, g, H, W)
    var = d2.mean(1, keepdim=True)                              # sigma_j^2 per group
    s = torch.exp(-d2 / (var + eps))
    if sal_b is not None:
        s = s * torch.stack([neighbour(sal_b.float(), dy, dx) for dy, dx in NEIGHBOURS], 1)
    return s.flatten(1, 2)


# --------------------------------------------------------------------------- #
# SFA: Structures-Features Attention (Sec. 3.3.3, Eqs. 5-6)
# --------------------------------------------------------------------------- #
class SFA(nn.Module):
    """
    Q from the distilled structure, K and V from the fused image features.
    DECISION (the paper's biggest gap): softmax 'across spatial locations' over ALL pixels would
    be a 65,536 x 65,536 matrix at 256x256 (impossible within the reported 4.6 GB and against the
    paper's linear-complexity claim). Default mode="window": attention inside non-overlapping
    window x window patches (linear in image size). mode="channel": transposed (d x d) attention,
    also linear. Multi-head, pre-LayerNorm and a residual connection (E + output) as in DCAMSR's
    attention blocks.
    """

    def __init__(self, d=64, heads=4, window=8, mode="window"):
        super().__init__()
        assert mode in ("window", "channel") and d % heads == 0
        self.d, self.heads, self.window, self.mode = d, heads, window, mode
        self.norm_q, self.norm_e = LayerNorm2d(d), LayerNorm2d(d)
        self.q = nn.Conv2d(d, d, 1)
        self.k = nn.Conv2d(d, d, 1)
        self.v = nn.Conv2d(d, d, 1)
        self.out = nn.Conv2d(d, d, 1)
        if mode == "channel":
            self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

    def forward(self, struct, feat):
        q = self.q(self.norm_q(struct))
        e = self.norm_e(feat)
        k, v = self.k(e), self.v(e)
        y = self._window(q, k, v) if self.mode == "window" else self._channel(q, k, v)
        return feat + self.out(y)

    def _window(self, q, k, v):
        B, C, H, W = q.shape
        w, h = self.window, self.heads
        ph, pw = (-H) % w, (-W) % w
        if ph or pw:
            q, k, v = (F.pad(t, (0, pw, 0, ph)) for t in (q, k, v))
        Hp, Wp = H + ph, W + pw

        def to_windows(t):   # (B,C,Hp,Wp) -> (B*nW, heads, w*w, C/heads)
            t = t.view(B, h, C // h, Hp // w, w, Wp // w, w)
            return t.permute(0, 3, 5, 1, 4, 6, 2).reshape(-1, h, w * w, C // h)

        y = F.scaled_dot_product_attention(to_windows(q), to_windows(k), to_windows(v))   # softmax(QK^T/sqrt(dk))V
        y = y.view(B, Hp // w, Wp // w, h, w, w, C // h).permute(0, 3, 6, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        return y[..., :H, :W]

    def _channel(self, q, k, v):
        B, C, H, W = q.shape
        h = self.heads
        q, k, v = (t.view(B, h, C // h, H * W) for t in (q, k, v))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        a = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(-1)
        return (a @ v).view(B, C, H, W)


# --------------------------------------------------------------------------- #
# MCSD block (Sec. 3.3, Fig. 1b)
# --------------------------------------------------------------------------- #
class MCSD(nn.Module):
    """
    Ablation switches (paper Fig. 5a): intra / cross similarities on or off, SAN on or off.
    With intra=cross=False the block is the 'MC baseline': fused features only, no structure query.
    """

    def __init__(self, d=64, groups=8, d_squeeze=64, heads=4, window=8, sfa_mode="window",
                 intra=True, cross=True, san=True, eps=1e-6):
        super().__init__()
        assert d % groups == 0
        self.groups, self.eps = groups, eps
        self.intra, self.cross, self.use_san = intra, cross, san
        # "fusion of the target and reference encoder features via concatenation and convolution"
        self.fuse = nn.Conv2d(2 * d, d, 3, 1, 1)
        n_pairs = 2 * int(intra) + 2 * int(cross)
        self.structural = n_pairs > 0
        if self.structural:
            if san:
                self.san_tar, self.san_ref = SAN(d, groups), SAN(d, groups)
            self.squeeze = nn.Conv2d(n_pairs * len(NEIGHBOURS) * groups, d_squeeze, 1)   # Eq. (4)
            self.sfa = SFA(d_squeeze, heads, window, sfa_mode) if d_squeeze == d else None
            assert self.sfa is not None, "d_squeeze must equal the feature width (paper: d' = 64)"

    def distilled(self, e_tar, e_ref):
        """Eq. (3): [S_tt*s_tar, S_tr*s_ref, S_rt*s_tar, S_rr*s_ref] (saliency of the NEIGHBOUR contrast)."""
        g, eps = self.groups, self.eps
        s_tar = self.san_tar(e_tar) if self.use_san else None
        s_ref = self.san_ref(e_ref) if self.use_san else None
        parts = []
        if self.intra:
            parts.append(neighbourhood_similarity(e_tar, e_tar, g, eps, s_tar))
        if self.cross:
            parts.append(neighbourhood_similarity(e_tar, e_ref, g, eps, s_ref))
            parts.append(neighbourhood_similarity(e_ref, e_tar, g, eps, s_tar))
        if self.intra:
            parts.append(neighbourhood_similarity(e_ref, e_ref, g, eps, s_ref))
        return torch.cat(parts, 1)

    def forward(self, e_tar, e_ref):
        fused = self.fuse(torch.cat([e_tar, e_ref], 1))
        if not self.structural:
            return fused
        squeezed = self.squeeze(self.distilled(e_tar, e_ref).to(fused.dtype))
        return self.sfa(squeezed, fused)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class SGFMRI(nn.Module):
    """
    Call signature model(lr, ref, ref_down) like DCAMSR. scale=1 = reconstruction.
    DECISION: ONE encoder shared by both contrasts, as in DCAMSR (the paper writes separate
    Phi_tar / Phi_ref, but only a shared encoder matches its '4M parameters': shared ~4.1 M,
    separate ~5.1 M).
    """

    def __init__(self, scale=4, nf=64, groups=8, heads=4, window=8, sfa_mode="window",
                 intra=True, cross=True, san=True, separate_encoders=False):
        super().__init__()
        self.scale = scale
        self.enc = Encoder(in_chl=1, nf=nf, n_blks=[4, 4, 4])
        self.enc_ref = Encoder(in_chl=1, nf=nf, n_blks=[4, 4, 4]) if separate_encoders else None
        self.decoder = Decoder(nf, 1, n_blks=[2, 2, 2, 12, 8, 4])
        self.mcsd = nn.ModuleList([MCSD(nf, groups, nf, heads, window, sfa_mode, intra, cross, san)
                                   for _ in range(3)])

    def forward(self, lr, ref, ref_down=None):
        lrsr = F.interpolate(lr, scale_factor=self.scale, mode="bilinear", align_corners=False) \
            if self.scale != 1 else lr
        f_tar = self.enc(lrsr)
        f_ref = (self.enc_ref or self.enc)(ref)
        fused = [m(t, r) for m, t, r in zip(self.mcsd, f_tar, f_ref)]
        return self.decoder(f_tar, fused) + lrsr