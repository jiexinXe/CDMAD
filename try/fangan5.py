# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP)
# Modified (Scheme-5): "theoretically-grounded" optimal baseline picture via PRIOR-MATCHING optimization
#
# (Diagnostics-added version)
#   Adds CSV diagnostics to help analyze why debias may appear ineffective:
#     - bias vector stats: std, range, mean, L2 norm
#     - argmax flip rate before/after debias (weak unlabeled in train; first batch in val)
#     - PMB optimizer metrics: KL / TV / L2 / ||b||
#
# Core idea (Scheme-5, standalone):
#   Choose a baseline picture I_b by matching the marginal prediction distribution (after debias) to a target prior π:
#       p_bar(b) = mean_u softmax( z_theta(u) - b )
#       b = z_theta(I_b)
#       minimize KL(p_bar(b) || π) + λ_tv TV(I_b) + λ_l2 ||I_b||^2
#
# Independence note:
#   This script is independent from Scheme-1/2/3/4.

from __future__ import print_function

import argparse
import math
import os
import shutil
import time
import random
import csv
from typing import Optional, Dict, Any

import numpy as np

# Restore np.int for legacy numpy usage
setattr(np, 'int', int)

import wrn as models
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p


# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser(description='PyTorch FixMatch Training (CDMAD baseline, Scheme-5 prior-matched baseline picture, with diagnostics CSV)')

# Optimization
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.0015, type=float)

# Checkpoints
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--out', default='result', type=str)

# Miscs
parser.add_argument('--manualSeed', type=int, default=0)
parser.add_argument('--gpu', default='0', type=str)

# Long-tail setup
parser.add_argument('--num_max', type=int, default=1500)
parser.add_argument('--num_max_u', type=int, default=3000)
parser.add_argument('--imb_ratio', type=int, default=100)
parser.add_argument('--imb_ratio_u', type=float, default=100)
parser.add_argument('--step', action='store_true')  # legacy
parser.add_argument('--val-iteration', type=int, default=500)

# FixMatch hyper-params
parser.add_argument('--tau', default=0, type=float, help='threshold for pseudo-label in FixMatch')
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--wd', default=0.04, type=float)

# Dataset / imbalance type
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10/cifar100/stl10')
parser.add_argument('--imbalancetype', type=str, default='long', help='long or step imbalance')
parser.add_argument('--unlabeledratio', type=float, default=2)
parser.add_argument('--debiasstart', type=int, default=100, help='epoch to start debias')

# -------------------------
# Scheme-5: Prior-matched baseline (PMB) options
# -------------------------
parser.add_argument('--pm-num-probes', type=int, default=1, help='number of baseline pictures (probes) to maintain')
parser.add_argument('--pm-update-freq', type=int, default=20, help='update baseline every N iterations (global)')
parser.add_argument('--pm-steps', type=int, default=3, help='gradient steps per baseline update')
parser.add_argument('--pm-lr', type=float, default=0.08, help='baseline optimizer lr')
parser.add_argument('--pm-warmup-iters', type=int, default=100, help='warmup iters before updating baseline')
parser.add_argument('--pm-mode', type=str, default='eval', choices=['eval', 'train'],
                    help='forward baseline under model.eval() (recommended) or model.train()')
parser.add_argument('--pm-clamp', type=float, default=4.0, help='clamp baseline tensor to [-pm-clamp, pm-clamp]')

# Target prior π
parser.add_argument('--pm-target', type=str, default='uniform', choices=['uniform', 'labeled'],
                    help='target prior π: uniform or labeled empirical prior (from N_SAMPLES_PER_CLASS)')
parser.add_argument('--pm-kl', type=str, default='forward', choices=['forward', 'reverse'],
                    help='KL direction: forward KL(p_bar||π) or reverse KL(π||p_bar)')

# Regularization (to keep probe non-semantic / smooth)
parser.add_argument('--pm-tv', type=float, default=1e-3, help='TV regularization weight')
parser.add_argument('--pm-l2', type=float, default=1e-4, help='L2 regularization weight')

# Optional: center bias logits (remove additive constant)
parser.add_argument('--pm-center', action='store_true', help='center bias logits by subtracting mean')

# If you want test-time debias to use PMB too (default yes)
parser.add_argument('--pm-test-debias', action='store_true', help='apply PMB debias at test time (default: enabled)')
parser.set_defaults(pm_test_debias=True)

# -------------------------
# Diagnostics
# -------------------------

# -------------------------
# Scheme-5B: Fourier-constrained baseline picture (fixed amplitude, learnable phase)
# -------------------------
parser.add_argument('--pm-param', type=str, default='fft', choices=['fft', 'pixel'],
                    help='parameterization of baseline picture: fft (fixed amplitude, learn phase) or pixel (direct pixels)')
parser.add_argument('--fft-amp-momentum', type=float, default=0.99,
                    help='EMA momentum for amplitude spectrum estimated from weak unlabeled batches')
parser.add_argument('--fft-amp-eps', type=float, default=1e-6,
                    help='epsilon to avoid zero amplitude')
parser.add_argument('--fft-phase-init', type=str, default='random', choices=['random', 'zero'],
                    help='phase initialization for FFT probes')
parser.add_argument('--fft-img-scale', type=float, default=2.0,
                    help='tanh scale for reconstructed probe images to avoid adversarial extremes')
parser.add_argument('--fft-img-center', action='store_true',
                    help='center reconstructed probe images per-channel (remove spatial mean)')
parser.set_defaults(fft_img_center=True)

parser.add_argument('--diag-enable', action='store_true', help='enable writing diagnostics CSV')
parser.add_argument('--diag-freq', type=int, default=50, help='log diagnostics every N training iters')
parser.add_argument('--diag-file', type=str, default='diagnostics.csv', help='diagnostics CSV filename inside --out')
parser.add_argument('--diag-val-first-batch', action='store_true', help='log val/test argmax flip rate on first batch')
parser.set_defaults(diag_val_first_batch=True)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# -------------------------
# Dataset import
# -------------------------
if args.dataset == 'cifar10':
    import dataset.fix_cifar10 as dataset
    print('==> Preparing imbalanced CIFAR10')
    num_class = 10
elif args.dataset == 'cifar100':
    import dataset.fix_cifar100 as dataset
    print('==> Preparing imbalanced CIFAR100')
    num_class = 100
    args.wd = 0.08
elif args.dataset == 'stl10':
    import dataset.fix_stl10 as dataset
    print('==> Preparing imbalanced STL_10')
    num_class = 10
else:
    raise ValueError(f'Unsupported dataset: {args.dataset}')

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Seeds
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)

random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -------------------------
# Diagnostics CSV
# -------------------------
class DiagnosticsCSV:
    """
    Lightweight CSV logger for debugging PMB effectiveness.
    """
    def __init__(self, out_dir: str, filename: str):
        self.path = os.path.join(out_dir, filename)
        self._fh = None
        self._writer = None
        self._header_written = False

    def open(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        file_exists = os.path.isfile(self.path) and os.path.getsize(self.path) > 0
        self._fh = open(self.path, 'a', newline='')
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames())
        if not file_exists:
            self._writer.writeheader()
            self._fh.flush()
        self._header_written = True

    @staticmethod
    def _fieldnames():
        return [
            # identifiers
            'ts', 'phase', 'epoch', 'iter', 'batch',
            # config
            'pm_target', 'pm_kl', 'debiasstart', 'tau',
            # PMB metrics
            'pmb_kl', 'pmb_tv', 'pmb_l2', 'pmb_bnorm',
            # bias stats
            'b_mean', 'b_std', 'b_range', 'b_norm',
            # probe stats (scheme-5B)
            'amp_mean', 'amp_std', 'phase_std',
            'probe_img_mean', 'probe_img_std', 'probe_img_range',
            # effect stats
            'pred_flip_rate', 'maxprob_mean_before', 'maxprob_mean_after',
            # test summary (optional)
            'test_top1', 'test_bacc', 'test_gm', 'test_bacc_debias', 'test_gm_debias'
        ]

    def log(self, row: Dict[str, Any]):
        if self._writer is None:
            return
        # fill missing with blanks
        base = {k: '' for k in self._fieldnames()}
        base.update(row)
        self._writer.writerow(base)
        self._fh.flush()

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
        self._fh = None
        self._writer = None


# -------------------------
# Scheme-5: Prior-matched baseline picture
# -------------------------
def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Isotropic TV on a batch of images: x [B,C,H,W]."""
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
    return (dh.abs().mean() + dw.abs().mean())



class FourierPriorMatchedBaseline:
    """
    Scheme-5B: Fourier-constrained baseline picture.

    Instead of optimizing pixels directly (which can easily become adversarial / unnatural),
    we parameterize probes in Fourier domain:
        - Amplitude spectrum A is estimated from weak unlabeled batches (EMA), and kept FIXED.
        - Phase Φ is learnable.
    Probe image is reconstructed by:
        I = irfft2( A * exp(iΦ) )

    We then perform the same prior-matching objective:
        p_bar = mean_u softmax( z(u) - b ),  b = mean_m z(I_m)
        L = KL(p_bar || π) + λ_tv TV(I) + λ_l2 ||I||^2
    """
    def __init__(self, device: torch.device, num_class: int):
        self.device = device
        self.num_class = num_class

        self.HW = None
        self.global_iter = 0

        # FFT params
        self.phase = None   # nn.Parameter [M, C, H, Wf]
        self.opt = None
        self.amp_ema = None # Tensor [C, H, Wf] (no grad)
        self.last_metrics = {}
        self.pi = None

    def set_target_prior(self, pi: np.ndarray):
        pi = np.asarray(pi, dtype=np.float32)
        pi = pi / max(1e-12, float(pi.sum()))
        self.pi = torch.tensor(pi, device=self.device).clamp(min=1e-8)

    def _ensure_shape(self, H: int, W: int):
        if self.HW == (H, W) and self.phase is not None:
            return
        self.HW = (H, W)
        M = max(2, int(args.pm_num_probes))  # enforce >=2 to avoid squeeze pitfalls
        Wf = W // 2 + 1

        if args.fft_phase_init == 'zero':
            init = torch.zeros((M, 3, H, Wf), device=self.device)
        else:
            # random phase in [-pi, pi]
            init = (torch.rand((M, 3, H, Wf), device=self.device) * 2.0 * math.pi) - math.pi

        self.phase = nn.Parameter(init)
        self.opt = optim.Adam([self.phase], lr=float(args.pm_lr))
        self.amp_ema = None

    @staticmethod
    def _wrap_phase(phi: torch.Tensor) -> torch.Tensor:
        # wrap to [-pi, pi] to keep phase bounded (helps stability)
        return ((phi + math.pi) % (2.0 * math.pi)) - math.pi

    def _update_amplitude(self, inputs_u_weak: torch.Tensor):
        """Estimate amplitude spectrum from a weak unlabeled batch, update EMA."""
        H, W = int(inputs_u_weak.shape[-2]), int(inputs_u_weak.shape[-1])
        self._ensure_shape(H, W)
        with torch.no_grad():
            spec = torch.fft.rfft2(inputs_u_weak, dim=(-2, -1))  # [B,C,H,Wf]
            amp = torch.abs(spec).mean(dim=0)                    # [C,H,Wf]
            amp = amp.clamp(min=float(args.fft_amp_eps))
            if self.amp_ema is None:
                self.amp_ema = amp.detach()
            else:
                m = float(args.fft_amp_momentum)
                self.amp_ema = (m * self.amp_ema) + ((1.0 - m) * amp.detach())

    def _reconstruct_probes(self) -> torch.Tensor:
        """Reconstruct probes from amp_ema and learnable phase."""
        H, W = self.HW
        if self.amp_ema is None:
            # fallback: unit amplitude (rare; only if called before any update_amp)
            Wf = W // 2 + 1
            self.amp_ema = torch.ones((3, H, Wf), device=self.device)

        amp = self.amp_ema.unsqueeze(0)  # [1,C,H,Wf]
        phi = self.phase
        # complex spectrum: amp * exp(i phi)
        spec = amp * torch.exp(1j * phi)
        img = torch.fft.irfft2(spec, s=(H, W), dim=(-2, -1)).real  # [M,C,H,W]

        if args.fft_img_center:
            img = img - img.mean(dim=(-2, -1), keepdim=True)

        # soft bounding in image space to avoid extreme adversarial patterns
        scale = float(args.fft_img_scale)
        if scale > 0:
            img = scale * torch.tanh(img / scale)

        return img

    @staticmethod
    def _center_logits(b: torch.Tensor) -> torch.Tensor:
        return b - b.mean()

    def _bias_from_probes(self, model: nn.Module):
        probes = self._reconstruct_probes()         # [M,3,H,W] with grad through phase
        logits_b, _ = model(probes)
        # robust shape
        if logits_b.dim() == 1:
            if logits_b.numel() == self.num_class:
                logits_b = logits_b.unsqueeze(0)
            else:
                raise RuntimeError(f"[FPMB] Unexpected logits_b shape {tuple(logits_b.shape)} for num_class={self.num_class}")
        elif logits_b.dim() != 2:
            raise RuntimeError(f"[FPMB] Unexpected logits_b dim={logits_b.dim()} shape={tuple(logits_b.shape)}")

        b = logits_b.mean(dim=0)  # [C]
        if args.pm_center:
            b = self._center_logits(b)

        if b.dim() != 1 or b.numel() != self.num_class:
            raise RuntimeError(f"[FPMB] Bias vector has wrong shape: {tuple(b.shape)}, expected ({self.num_class},)")

        return b, probes

    def maybe_update(self, model: nn.Module, inputs_u_weak: torch.Tensor):
        """
        Periodically update phase parameters by prior matching on weak unlabeled batch.
        """
        self.global_iter += 1
        B, C, H, W = inputs_u_weak.shape
        self._ensure_shape(H, W)

        # always refresh amplitude EMA (cheap)
        self._update_amplitude(inputs_u_weak)

        if self.pi is None:
            self.set_target_prior(np.ones(self.num_class, dtype=np.float32))

        if self.global_iter < int(args.pm_warmup_iters):
            return
        if int(args.pm_update_freq) > 1 and (self.global_iter % int(args.pm_update_freq) != 0):
            return

        prev_train = model.training
        if args.pm_mode == 'eval':
            model.eval()
        else:
            model.train()

        with torch.no_grad():
            logits_u, _ = model(inputs_u_weak)  # [B,C]

        kl_vals, tv_vals, l2_vals, bnorm_vals = [], [], [], []

        for _ in range(max(1, int(args.pm_steps))):
            self.opt.zero_grad(set_to_none=True)

            b, probes = self._bias_from_probes(model)
            bnorm_vals.append(float(b.detach().norm(p=2).item()))

            debiased = logits_u - b.view(1, -1)
            p_bar = F.softmax(debiased, dim=1).mean(dim=0).clamp(min=1e-8)

            pi = self.pi
            if args.pm_kl == 'forward':
                kl = torch.sum(p_bar * (torch.log(p_bar) - torch.log(pi)))
            else:
                kl = torch.sum(pi * (torch.log(pi) - torch.log(p_bar)))

            tv = total_variation(probes)
            l2 = torch.mean(probes * probes)

            loss = kl + float(args.pm_tv) * tv + float(args.pm_l2) * l2
            loss.backward()
            self.opt.step()

            with torch.no_grad():
                self.phase.copy_(self._wrap_phase(self.phase))

            kl_vals.append(float(kl.detach().item()))
            tv_vals.append(float(tv.detach().item()))
            l2_vals.append(float(l2.detach().item()))

        model.train(prev_train)

        self.last_metrics = {
            'kl': float(np.mean(kl_vals)) if kl_vals else 0.0,
            'tv': float(np.mean(tv_vals)) if tv_vals else 0.0,
            'l2': float(np.mean(l2_vals)) if l2_vals else 0.0,
            'bnorm': float(np.mean(bnorm_vals)) if bnorm_vals else 0.0,
        }

    @torch.no_grad()
    def bias_logits(self, model: nn.Module, H: int, W: int) -> torch.Tensor:
        """Return b in R^C computed from current probes."""
        self._ensure_shape(H, W)
        probes = self._reconstruct_probes().detach()

        prev_train = model.training
        model.eval()
        logits_b, _ = model(probes)
        model.train(prev_train)

        if logits_b.dim() == 1:
            if logits_b.numel() == self.num_class:
                logits_b = logits_b.unsqueeze(0)
            else:
                raise RuntimeError(f"[FPMB] Unexpected logits_b shape {tuple(logits_b.shape)} for num_class={self.num_class}")
        elif logits_b.dim() != 2:
            raise RuntimeError(f"[FPMB] Unexpected logits_b dim={logits_b.dim()} shape={tuple(logits_b.shape)}")

        b = logits_b.mean(dim=0)
        if args.pm_center:
            b = self._center_logits(b)
        if b.dim() != 1 or b.numel() != self.num_class:
            raise RuntimeError(f"[FPMB] Bias vector wrong shape {tuple(b.shape)} expected ({self.num_class},)")
        return b.detach()

    @torch.no_grad()
    def extra_diag_stats(self) -> dict:
        """Extra stats for diagnostics CSV."""
        out = {}
        if self.amp_ema is not None:
            out['amp_mean'] = float(self.amp_ema.mean().item())
            out['amp_std'] = float(self.amp_ema.std(unbiased=False).item())
        else:
            out['amp_mean'] = ''
            out['amp_std'] = ''

        if self.phase is not None:
            out['phase_std'] = float(self.phase.std(unbiased=False).item())
        else:
            out['phase_std'] = ''

        if self.HW is not None and self.phase is not None:
            probes = self._reconstruct_probes().detach()
            out['probe_img_mean'] = float(probes.mean().item())
            out['probe_img_std'] = float(probes.std(unbiased=False).item())
            out['probe_img_range'] = float((probes.max() - probes.min()).item())
        else:
            out['probe_img_mean'] = ''
            out['probe_img_std'] = ''
            out['probe_img_range'] = ''
        return out


class PriorMatchedBaseline:
    """
    Backward-compatible wrapper:
      - pm-param=pixel: original pixel-parameterized PMB
      - pm-param=fft:   Scheme-5B FourierPriorMatchedBaseline (recommended)
    """
    def __init__(self, device: torch.device, num_class: int):
        self.device = device
        self.num_class = num_class
        if args.pm_param == 'pixel':
            self.impl = _PixelPriorMatchedBaseline(device=device, num_class=num_class)
        else:
            self.impl = FourierPriorMatchedBaseline(device=device, num_class=num_class)

    def set_target_prior(self, pi: np.ndarray):
        return self.impl.set_target_prior(pi)

    @property
    def last_metrics(self):
        return self.impl.last_metrics

    @property
    def global_iter(self):
        return self.impl.global_iter

    @global_iter.setter
    def global_iter(self, v):
        self.impl.global_iter = v

    def maybe_update(self, model: nn.Module, inputs_u_weak: torch.Tensor):
        return self.impl.maybe_update(model, inputs_u_weak)

    def bias_logits(self, model: nn.Module, H: int, W: int) -> torch.Tensor:
        return self.impl.bias_logits(model, H, W)

    def extra_diag_stats(self) -> dict:
        if hasattr(self.impl, 'extra_diag_stats'):
            return self.impl.extra_diag_stats()
        return {}

    # for checkpointing
    def state_dict(self) -> dict:
        sd = {'pm_param': args.pm_param}
        if isinstance(self.impl, FourierPriorMatchedBaseline):
            sd.update({
                'phase': self.impl.phase.detach().cpu() if self.impl.phase is not None else None,
                'amp_ema': self.impl.amp_ema.detach().cpu() if self.impl.amp_ema is not None else None,
                'HW': self.impl.HW,
                'global_iter': int(self.impl.global_iter),
            })
        else:
            sd.update({
                'probes': self.impl.probes.detach().cpu() if self.impl.probes is not None else None,
                'HW': self.impl.HW,
                'global_iter': int(self.impl.global_iter),
            })
        return sd

    def load_state_dict(self, sd: dict):
        # load global_iter if present
        if 'global_iter' in sd:
            self.impl.global_iter = int(sd['global_iter'])

        if isinstance(self.impl, FourierPriorMatchedBaseline):
            phase = sd.get('phase', None)
            amp_ema = sd.get('amp_ema', None)
            HW = sd.get('HW', None)
            if phase is not None and HW is not None:
                H, W = int(HW[0]), int(HW[1])
                self.impl._ensure_shape(H, W)
                with torch.no_grad():
                    self.impl.phase.copy_(phase.to(self.device))
                if amp_ema is not None:
                    self.impl.amp_ema = amp_ema.to(self.device)
        else:
            probes = sd.get('probes', None)
            HW = sd.get('HW', None)
            if probes is not None and HW is not None:
                H, W = int(HW[0]), int(HW[1])
                self.impl._ensure_shape(H, W)
                with torch.no_grad():
                    self.impl.probes.copy_(probes.to(self.device))


class _PixelPriorMatchedBaseline:
    """
    Original pixel-parameterized PMB (kept for fallback / ablations).
    """
    def __init__(self, device: torch.device, num_class: int):
        self.device = device
        self.num_class = num_class
        self.probes = None  # nn.Parameter [M,3,H,W]
        self.opt = None
        self.HW = None
        self.global_iter = 0
        self.last_metrics = {}
        self.pi = None  # [C]

    def set_target_prior(self, pi: np.ndarray):
        pi = np.asarray(pi, dtype=np.float32)
        pi = pi / max(1e-12, float(pi.sum()))
        self.pi = torch.tensor(pi, device=self.device).clamp(min=1e-8)

    def _ensure_shape(self, H: int, W: int):
        if self.probes is not None and self.HW == (H, W):
            return
        M = max(1, int(args.pm_num_probes))
        init = torch.zeros((M, 3, H, W), device=self.device)
        init += 0.05 * torch.randn_like(init)
        self.probes = nn.Parameter(init)
        self.opt = optim.Adam([self.probes], lr=float(args.pm_lr))
        self.HW = (H, W)

    @staticmethod
    def _center_logits(b: torch.Tensor) -> torch.Tensor:
        return b - b.mean()

    def _bias_from_probes(self, model: nn.Module) -> torch.Tensor:
        logits_b, _ = model(self.probes)  # expected [M,C]
        if logits_b.dim() == 1:
            if logits_b.numel() == self.num_class:
                logits_b = logits_b.unsqueeze(0)
            else:
                raise RuntimeError(f"[PMB] Unexpected logits_b shape {tuple(logits_b.shape)} for num_class={self.num_class}")
        elif logits_b.dim() != 2:
            raise RuntimeError(f"[PMB] Unexpected logits_b dim={logits_b.dim()} shape={tuple(logits_b.shape)}")

        b = logits_b.mean(dim=0)
        if args.pm_center:
            b = self._center_logits(b)
        if b.dim() != 1 or b.numel() != self.num_class:
            raise RuntimeError(f"[PMB] Bias vector wrong shape {tuple(b.shape)} expected ({self.num_class},)")
        return b

    def maybe_update(self, model: nn.Module, inputs_u_weak: torch.Tensor):
        self.global_iter += 1
        B, C, H, W = inputs_u_weak.shape
        self._ensure_shape(H, W)

        if self.pi is None:
            self.set_target_prior(np.ones(self.num_class, dtype=np.float32))

        if self.global_iter < int(args.pm_warmup_iters):
            return
        if int(args.pm_update_freq) > 1 and (self.global_iter % int(args.pm_update_freq) != 0):
            return

        prev_train = model.training
        if args.pm_mode == 'eval':
            model.eval()
        else:
            model.train()

        with torch.no_grad():
            logits_u, _ = model(inputs_u_weak)

        kl_vals, tv_vals, l2_vals, bnorm_vals = [], [], [], []
        for _ in range(max(1, int(args.pm_steps))):
            self.opt.zero_grad(set_to_none=True)
            b = self._bias_from_probes(model)
            bnorm_vals.append(float(b.detach().norm(p=2).item()))

            debiased = logits_u - b.view(1, -1)
            p_bar = F.softmax(debiased, dim=1).mean(dim=0).clamp(min=1e-8)

            pi = self.pi
            if args.pm_kl == 'forward':
                kl = torch.sum(p_bar * (torch.log(p_bar) - torch.log(pi)))
            else:
                kl = torch.sum(pi * (torch.log(pi) - torch.log(p_bar)))

            tv = total_variation(self.probes)
            l2 = torch.mean(self.probes * self.probes)
            loss = kl + float(args.pm_tv) * tv + float(args.pm_l2) * l2
            loss.backward()
            self.opt.step()

            clamp_v = float(args.pm_clamp)
            if clamp_v > 0:
                with torch.no_grad():
                    self.probes.clamp_(min=-clamp_v, max=clamp_v)

            kl_vals.append(float(kl.detach().item()))
            tv_vals.append(float(tv.detach().item()))
            l2_vals.append(float(l2.detach().item()))

        model.train(prev_train)

        self.last_metrics = {
            'kl': float(np.mean(kl_vals)) if kl_vals else 0.0,
            'tv': float(np.mean(tv_vals)) if tv_vals else 0.0,
            'l2': float(np.mean(l2_vals)) if l2_vals else 0.0,
            'bnorm': float(np.mean(bnorm_vals)) if bnorm_vals else 0.0,
        }

    @torch.no_grad()
    def bias_logits(self, model: nn.Module, H: int, W: int) -> torch.Tensor:
        self._ensure_shape(H, W)
        prev_train = model.training
        model.eval()
        logits_b, _ = model(self.probes.detach())
        model.train(prev_train)

        if logits_b.dim() == 1:
            if logits_b.numel() == self.num_class:
                logits_b = logits_b.unsqueeze(0)
            else:
                raise RuntimeError(f"[PMB] Unexpected logits_b shape {tuple(logits_b.shape)} for num_class={self.num_class}")
        elif logits_b.dim() != 2:
            raise RuntimeError(f"[PMB] Unexpected logits_b dim={logits_b.dim()} shape={tuple(logits_b.shape)}")

        b = logits_b.mean(dim=0)
        if args.pm_center:
            b = self._center_logits(b)
        if b.dim() != 1 or b.numel() != self.num_class:
            raise RuntimeError(f"[PMB] Bias vector wrong shape {tuple(b.shape)} expected ({self.num_class},)")
        return b.detach()

# -------------------------
# Main
# -------------------------
def main():
    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    diag: Optional[DiagnosticsCSV] = None
    if args.diag_enable:
        diag = DiagnosticsCSV(args.out, args.diag_file)
        diag.open()

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio, args.imbalancetype)
    U_SAMPLES_PER_CLASS = make_imb_data(args.num_max_u, num_class, args.imb_ratio_u, args.imbalancetype)

    if np.array(N_SAMPLES_PER_CLASS).sum() + np.array(U_SAMPLES_PER_CLASS).sum() >= 30000 or args.dataset == 'stl10':
        args.wd = 0.01

    if args.dataset == 'cifar10':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_cifar10(
            './data', N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS, rand_number=args.manualSeed
        )
    elif args.dataset == 'stl10':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_stl10(
            './data', N_SAMPLES_PER_CLASS, args.out, rand_number=args.manualSeed
        )
    elif args.dataset == 'cifar100':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_cifar100(
            './data', N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS, rand_number=args.manualSeed
        )
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=int(args.unlabeledratio * args.batch_size),
                                            shuffle=True, num_workers=4, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=200, shuffle=False, num_workers=4)

    print("==> creating WRN-28-2")

    def create_model(ema=False):
        model = models.WRN(2, num_classes=num_class).cuda()
        params = list(model.parameters())
        if ema:
            for p in params:
                p.detach_()
        return model, params

    model, params = create_model()
    ema_model, _ = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in params) / 1e6))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(params, lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pmb = PriorMatchedBaseline(device=device, num_class=num_class)

    if args.pm_target == 'uniform':
        pi = np.ones(num_class, dtype=np.float32)
    else:
        pi = np.asarray(N_SAMPLES_PER_CLASS, dtype=np.float32)
    pmb.set_target_prior(pi)

    title = 'fixcdmad-scheme5-diag-' + args.dataset
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'pmb_state' in checkpoint and checkpoint['pmb_state'] is not None:
            pmb.load_state_dict(checkpoint['pmb_state'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['bACC', 'GM', 'bACC_debias', 'GM_debias', 'Top1'])

    global_iter = 0
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f | PMTarget: %s | KL: %s' %
              (epoch + 1, args.epochs, state['lr'], args.pm_target, args.pm_kl))

        global_iter = train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
                            ema_optimizer, train_criterion, epoch, pmb, global_iter, diag)

        test_acc1, testclassacc1, test_acc2, testclassacc2, test_flip = validate(
            test_loader, ema_model, criterion, mode='Test Stats ', epoch=epoch, pmb=pmb, diag=diag, global_iter=global_iter
        )

        GM = geometric_mean(testclassacc1)
        GM2 = geometric_mean(testclassacc2)

        print("without test debias bACC:", testclassacc1.mean(), "GM:", GM,
              "with test debias bACC:", testclassacc2.mean(), "GM:", GM2)

        logger.append([testclassacc1.mean(), GM, testclassacc2.mean(), GM2, test_acc1])

        if diag is not None:
            diag.log({
                'ts': time.time(),
                'phase': 'epoch_end',
                'epoch': epoch,
                'iter': global_iter,
                'batch': '',
                'pm_target': args.pm_target,
                'pm_kl': args.pm_kl,
                'debiasstart': args.debiasstart,
                'tau': args.tau,
                'test_top1': float(test_acc1),
                'test_bacc': float(testclassacc1.mean()),
                'test_gm': float(GM),
                'test_bacc_debias': float(testclassacc2.mean()),
                'test_gm_debias': float(GM2),
                **pmb.extra_diag_stats(),
                'pred_flip_rate': float(test_flip) if test_flip is not None else '',
            })

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pmb_state': pmb.state_dict(),
        }, epoch + 1)

    logger.close()
    if diag is not None:
        diag.close()


def geometric_mean(accperclass: np.ndarray) -> float:
    gm = 1.0
    for i in range(num_class):
        if accperclass[i] == 0:
            gm *= (1 / (100 * num_class)) ** (1 / num_class)
        else:
            gm *= (accperclass[i]) ** (1 / num_class)
    return float(gm)


@torch.no_grad()
def _bias_stats(b: torch.Tensor):
    b = b.detach().float()
    b_mean = float(b.mean().item())
    b_std = float(b.std(unbiased=False).item())
    b_range = float((b.max() - b.min()).item())
    b_norm = float(b.norm(p=2).item())
    return b_mean, b_std, b_range, b_norm


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion,
          epoch: int, pmb: PriorMatchedBaseline, global_iter: int, diag: Optional[DiagnosticsCSV]):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()

    kl_m = AverageMeter()
    tv_m = AverageMeter()
    bnorm_m = AverageMeter()

    end = time.time()
    bar = Bar('Training', max=args.val_iteration)

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()

    for batch_idx in range(args.val_iteration):
        global_iter += 1

        try:
            inputs_x, targets_x, _ = next(labeled_train_iter)
        except Exception:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = next(unlabeled_train_iter)
        except Exception:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = next(unlabeled_train_iter)

        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        targets_x2 = torch.zeros(batch_size, num_class).scatter_(1, targets_x.view(-1, 1), 1)

        inputs_x, targets_x2 = inputs_x.cuda(), targets_x2.cuda(non_blocking=True)
        inputs_u, inputs_u2, inputs_u3 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()

        H, W = int(inputs_x.shape[2]), int(inputs_x.shape[3])

        # Update baseline probes (only probes updated)
        with torch.enable_grad():
            pmb.maybe_update(model, inputs_u)

        if pmb.last_metrics:
            kl_m.update(pmb.last_metrics.get('kl', 0.0), 1)
            tv_m.update(pmb.last_metrics.get('tv', 0.0), 1)
            bnorm_m.update(pmb.last_metrics.get('bnorm', 0.0), 1)

        with torch.no_grad():
            outputs_u_raw, _ = model(inputs_u)

            debias_on = int(epoch > args.debiasstart)
            if debias_on:
                biaseddegree = pmb.bias_logits(model, H, W)  # [C]
                outputs_u = outputs_u_raw - biaseddegree.view(1, -1)
            else:
                biaseddegree = torch.zeros(num_class, device=outputs_u_raw.device)
                outputs_u = outputs_u_raw

            targets_u2 = F.softmax(outputs_u, dim=1).detach()

        # --- Diagnostics (train) ---
        if diag is not None and (global_iter % max(1, int(args.diag_freq)) == 0):
            with torch.no_grad():
                pred_before = torch.argmax(outputs_u_raw, dim=1)
                pred_after = torch.argmax(outputs_u, dim=1)
                flip = float((pred_before != pred_after).float().mean().item())

                maxprob_before = float(F.softmax(outputs_u_raw, dim=1).max(dim=1)[0].mean().item())
                maxprob_after = float(F.softmax(outputs_u, dim=1).max(dim=1)[0].mean().item())

                b_mean, b_std, b_range, b_norm = _bias_stats(biaseddegree)

                diag.log({
                    'ts': time.time(),
                    'phase': 'train',
                    'epoch': epoch,
                    'iter': global_iter,
                    'batch': batch_idx,
                    'pm_target': args.pm_target,
                    'pm_kl': args.pm_kl,
                    'debiasstart': args.debiasstart,
                    'tau': args.tau,
                    'pmb_kl': float(pmb.last_metrics.get('kl', '')) if pmb.last_metrics else '',
                    'pmb_tv': float(pmb.last_metrics.get('tv', '')) if pmb.last_metrics else '',
                    'pmb_l2': float(pmb.last_metrics.get('l2', '')) if pmb.last_metrics else '',
                    'pmb_bnorm': float(pmb.last_metrics.get('bnorm', '')) if pmb.last_metrics else '',
                    'b_mean': b_mean,
                    'b_std': b_std,
                    'b_range': b_range,
                    'b_norm': b_norm,
                    'pred_flip_rate': flip,
                    'maxprob_mean_before': maxprob_before,
                    'maxprob_mean_after': maxprob_after,
                    **pmb.extra_diag_stats(),
                })

        max_p, p_hat = torch.max(targets_u2, dim=1)
        p_hat = torch.zeros(int(args.unlabeledratio * batch_size), num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
        select_mask = max_p.ge(args.tau)
        select_mask = torch.cat([select_mask, select_mask], 0).float()

        all_targets = torch.cat([targets_x2, targets_u2, targets_u2], dim=0)

        logits_x, _ = model(inputs_x)
        logits_u2, _ = model(inputs_u2)
        logits_u3, _ = model(inputs_u3)
        logits_u = torch.cat([logits_u2, logits_u3], dim=0)

        Lx, Lu = criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)
        loss = Lx + Lu

        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | ETA: {eta:} | ' \
                 'Loss: {loss:.4f} | Lx: {loss_x:.4f} | Lu: {loss_u:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                 )
        if kl_m.count > 0:
            suffix += ' | PMB_KL: %.4f TV: %.4f ||b||: %.3f' % (kl_m.avg, tv_m.avg, bnorm_m.avg)

        bar.suffix = suffix
        bar.next()

    bar.finish()
    return global_iter


def validate(valloader, model, criterion, mode: str, epoch: int, pmb: PriorMatchedBaseline,
             diag: Optional[DiagnosticsCSV], global_iter: int):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1debias = AverageMeter()
    top5debias = AverageMeter()

    model.eval()

    accperclass = np.zeros((num_class))
    accperclass2 = np.zeros((num_class))

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    pred_flip_first = None

    with torch.no_grad():
        first_batch_logged = False
        first_H = None
        first_W = None

        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            if first_H is None:
                first_H, first_W = int(inputs.shape[2]), int(inputs.shape[3])
                if args.pm_test_debias:
                    biaseddegree = pmb.bias_logits(model, first_H, first_W)
                else:
                    biaseddegree = torch.zeros(num_class, device=inputs.device)

                # Diagnostics (val): bias stats + flip on first batch
                if diag is not None and args.diag_val_first_batch:
                    outputs0, _ = model(inputs)
                    outputs1 = outputs0 - biaseddegree.view(1, -1)
                    pred0 = torch.argmax(outputs0, dim=1)
                    pred1 = torch.argmax(outputs1, dim=1)
                    pred_flip_first = float((pred0 != pred1).float().mean().item())

                    b_mean, b_std, b_range, b_norm = _bias_stats(biaseddegree)
                    diag.log({
                        'ts': time.time(),
                        'phase': 'val',
                        'epoch': epoch,
                        'iter': global_iter,
                        'batch': batch_idx,
                        'pm_target': args.pm_target,
                        'pm_kl': args.pm_kl,
                        'debiasstart': args.debiasstart,
                        'tau': args.tau,
                        'pmb_kl': float(pmb.last_metrics.get('kl', '')) if pmb.last_metrics else '',
                        'pmb_tv': float(pmb.last_metrics.get('tv', '')) if pmb.last_metrics else '',
                        'pmb_l2': float(pmb.last_metrics.get('l2', '')) if pmb.last_metrics else '',
                        'pmb_bnorm': float(pmb.last_metrics.get('bnorm', '')) if pmb.last_metrics else '',
                        'b_mean': b_mean,
                        'b_std': b_std,
                        'b_range': b_range,
                        'b_norm': b_norm,
                        'pred_flip_rate': pred_flip_first,
                        'maxprob_mean_before': float(F.softmax(outputs0, dim=1).max(dim=1)[0].mean().item()),
                        'maxprob_mean_after': float(F.softmax(outputs1, dim=1).max(dim=1)[0].mean().item()),
                    })

            outputs, _ = model(inputs)
            outputs2 = outputs - biaseddegree.view(1, -1)

            score = F.softmax(outputs, dim=1)
            score2 = F.softmax(outputs2, dim=1)

            prediction = torch.argmax(score, dim=1)
            prediction2 = torch.argmax(score2, dim=1)

            targetsonehot = torch.zeros(inputs.size(0), num_class).scatter_(1, targets.cpu().view(-1, 1).long(), 1)
            outputs2onehot = torch.zeros(inputs.size(0), num_class).scatter_(1, prediction.cpu().view(-1, 1).long(), 1)
            outputs2onehot2 = torch.zeros(inputs.size(0), num_class).scatter_(1, prediction2.cpu().view(-1, 1).long(), 1)

            accperclass = accperclass + torch.sum(targetsonehot * outputs2onehot, dim=0).cpu().numpy().astype(np.int64)
            accperclass2 = accperclass2 + torch.sum(targetsonehot * outputs2onehot2, dim=0).cpu().numpy().astype(np.int64)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            prec1debias, prec5debias = accuracy(outputs2, targets, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            top1debias.update(prec1debias.item(), inputs.size(0))
            top5debias.update(prec5debias.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | top1: {top1:.4f} | top5: {top5:.4f}'.format(
                batch=batch_idx + 1,
                size=len(valloader),
                data=data_time.avg,
                bt=batch_time.avg,
                top1=top1.avg,
                top5=top5.avg
            )
            bar.next()
        bar.finish()

    if args.dataset == 'cifar10':
        accperclass = accperclass / 1000
        accperclass2 = accperclass2 / 1000
    elif args.dataset == 'stl10':
        accperclass = accperclass / 800
        accperclass2 = accperclass2 / 800
    elif args.dataset == 'cifar100':
        accperclass = accperclass / 100
        accperclass2 = accperclass2 / 100

    return (top1.avg, accperclass, top1debias.avg, accperclass2, pred_flip_first)


def make_imb_data(max_num, class_num, gamma, imb):
    if imb == 'long':
        mu = np.power(1 / gamma, 1 / (class_num - 1))
        class_num_list = []
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(int(max_num / gamma))
            else:
                class_num_list.append(int(max_num * np.power(mu, i)))
        print(class_num_list)
    elif imb == 'step':
        class_num_list = []
        for i in range(class_num):
            if i < int(class_num / 2):
                class_num_list.append(int(max_num))
            else:
                class_num_list.append(int(max_num / gamma))
        print(class_num_list)
    else:
        raise ValueError(f'Unsupported imbalancetype: {imb}')
    return list(class_num_list)


def save_checkpoint(state, epoch, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if epoch % 100 == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_' + str(epoch) + '.pth.tar'))


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask):
        Lx = -torch.mean(torch.sum(torch.log(F.softmax(outputs_x, dim=1) + 1e-8) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(torch.log(F.softmax(outputs_u, dim=1) + 1e-8) * targets_u, dim=1) * mask)
        return Lx, Lu


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = args.wd * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.float()
            param = param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            param.mul_(1 - self.wd)


if __name__ == '__main__':
    main()
