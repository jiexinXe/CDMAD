# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP)
# Modified (Scheme-3): Frequency-domain controllable baseline picture (spectral random-phase probe family)
#
# Goal:
#   Replace CDMAD's single white/solid reference with a *frequency-domain* probe family that:
#     - remains semantic-irrelevant via random phase
#     - becomes statistically more representative by matching the dataset's average amplitude spectrum
#   while being controllable by a small set of parameters:
#     - --spec-lambda : representativeness strength (0 = DC-only constant, 1 = full amplitude-matched)
#     - --spec-band   : which frequency band to keep from the amplitude spectrum (low/mid/high/all)
#     - --spec-cutoff / --spec-mid-low / --spec-mid-high : band boundaries
#
# Independence note:
#   This script is intentionally independent from Scheme-1 and Scheme-2:
#     - no probe distributions/multi-family mixtures beyond the spectral family itself
#     - no bias decomposition b0/Δb, no ramp/EMA beyond maintaining spectral statistics

from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np

# Restore np.int for legacy numpy code
setattr(np, 'int', int)

import wrn as models
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p


# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser(description='PyTorch FixMatch Training (CDMAD baseline, Scheme-3 spectral probe)')

# Optimization options
parser.add_argument('--epochs', default=500, type=int, metavar='N')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.0015, type=float, metavar='LR')

# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--out', default='result', type=str)

# Miscs
parser.add_argument('--manualSeed', type=int, default=0)
parser.add_argument('--gpu', default='0', type=str)

# Long-tail setup
parser.add_argument('--num_max', type=int, default=1500)
parser.add_argument('--num_max_u', type=int, default=3000)
parser.add_argument('--imb_ratio', type=int, default=100)
parser.add_argument('--imb_ratio_u', type=float, default=100)
parser.add_argument('--step', action='store_true')  # legacy, kept for compatibility
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
# Scheme-3: Spectral probe options
# -------------------------
parser.add_argument('--spec-lambda', type=float, default=1.0,
                    help='representativeness strength: 0=DC-only (constant), 1=amplitude-matched (random phase)')
parser.add_argument('--spec-band', type=str, default='all', choices=['all', 'low', 'mid', 'high'],
                    help='frequency band to keep from the data amplitude spectrum')
parser.add_argument('--spec-cutoff', type=float, default=0.25,
                    help='cutoff radius in [0,1] for low/high (normalized by Nyquist). For mid, used as default low bound.')
parser.add_argument('--spec-mid-low', type=float, default=0.20,
                    help='mid-band low bound in [0,1] (only used when spec-band=mid)')
parser.add_argument('--spec-mid-high', type=float, default=0.60,
                    help='mid-band high bound in [0,1] (only used when spec-band=mid)')
parser.add_argument('--spec-k', type=int, default=8,
                    help='number of random-phase probes to average for bias logits (reduces variance)')
parser.add_argument('--spec-stat-ema', type=float, default=0.99,
                    help='EMA for running data amplitude spectrum')
parser.add_argument('--spec-clamp', type=float, default=4.0,
                    help='clamp generated probe tensor to [-spec-clamp, spec-clamp]')
parser.add_argument('--spec-mode', type=str, default='eval', choices=['eval', 'train'],
                    help='forward probes under model.eval() (recommended) or model.train()')
parser.add_argument('--spec-warmup-iters', type=int, default=200,
                    help='iterations to warm up amplitude spectrum stats before using amplitude-matched probes (fallback to DC-only)')

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

# -------------------------
# Environment / seeds
# -------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)

random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -------------------------
# Scheme-3: Spectral probe state
# -------------------------
class SpectralProbeState:
    """
    Maintains running average of data amplitude spectrum in rfft2 space:
      A_data: [1,C,H,W//2+1]
    Generates random-phase probes with controllable band and representativeness.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.A_data = None
        self.inited = False
        self.update_count = 0
        self._mask_cache = {}  # (H,W,band,cutoff,midlow,midhigh) -> mask [1,1,H,W//2+1]

    @staticmethod
    def _rfft_amp(x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] (real)
        X = torch.fft.rfft2(x.float(), dim=(-2, -1), norm='ortho')  # [B,C,H,W//2+1] complex
        return torch.abs(X)

    @torch.no_grad()
    def update_from_batch(self, x: torch.Tensor):
        """
        Update running amplitude spectrum stats from a batch of inputs in model input space.
        x: [B,C,H,W]
        """
        if x is None:
            return
        amp = self._rfft_amp(x).mean(dim=0, keepdim=True)  # [1,C,H,W//2+1]
        if not self.inited:
            self.A_data = amp.detach().clone()
            self.inited = True
            self.update_count = 1
            return
        m = float(args.spec_stat_ema)
        self.A_data.mul_(m).add_(amp * (1.0 - m))
        self.update_count += 1

    def _band_mask(self, H: int, W: int, band: str) -> torch.Tensor:
        """
        Build a radial mask in rfft2 space. For rfft2, W axis is 0..W//2.
        We define vertical freq index as k = min(i, H-i), horizontal l=j.
        Radius r normalized by Nyquist: r = sqrt((k/(H/2))^2 + (l/(W/2))^2).
        Returns mask shape [1,1,H,W//2+1] with 0/1 values.
        """
        key = (H, W, band, float(args.spec_cutoff), float(args.spec_mid_low), float(args.spec_mid_high))
        if key in self._mask_cache:
            return self._mask_cache[key]

        Wr = W // 2 + 1
        i = torch.arange(H, device=self.device)
        j = torch.arange(Wr, device=self.device)

        # vertical frequency magnitude: min(i, H-i)
        k = torch.minimum(i, H - i).float()
        l = j.float()

        # normalize by Nyquist (H/2, W/2)
        denom_k = max(1.0, float(H) / 2.0)
        denom_l = max(1.0, float(W) / 2.0)

        kk = (k / denom_k).view(H, 1).expand(H, Wr)
        ll = (l / denom_l).view(1, Wr).expand(H, Wr)
        r = torch.sqrt(kk * kk + ll * ll)  # [H,Wr] in [0, sqrt(2)]

        if band == 'all':
            mask = torch.ones_like(r)
        elif band == 'low':
            cutoff = float(args.spec_cutoff)
            mask = (r <= cutoff).float()
        elif band == 'high':
            cutoff = float(args.spec_cutoff)
            mask = (r >= cutoff).float()
        elif band == 'mid':
            lo = float(args.spec_mid_low)
            hi = float(args.spec_mid_high)
            if hi <= lo:
                # fallback: use cutoff and 2*cutoff
                lo = float(args.spec_cutoff)
                hi = min(1.0, 2.0 * lo)
            mask = ((r >= lo) & (r <= hi)).float()
        else:
            raise ValueError(f'Unsupported band: {band}')

        mask = mask.view(1, 1, H, Wr)
        self._mask_cache[key] = mask
        return mask

    @torch.no_grad()
    def sample_probe(self, H: int, W: int, K: int) -> torch.Tensor:
        """
        Generate K spectral probes: random phase + amplitude mix.
        Returns [K,C,H,W] in model input space.

        Amplitude:
          A_dc : DC-only amplitude (only (0,0) kept from A_data)
          A_band : A_data masked to selected band
          A_mix = (1-lambda) * A_dc + lambda * A_band
        """
        C = 3
        if (not self.inited) or (self.update_count < int(args.spec_warmup_iters)):
            # fallback to DC-only: constant image (all ones)
            return torch.ones((K, C, H, W), device=self.device)

        A = self.A_data  # [1,C,H,Wr]
        _, C, Hs, Wr = A.shape
        if Hs != H or Wr != (W // 2 + 1):
            # size mismatch: rebuild from zeros (rare if STL and CIFAR mixed)
            # For safety, fall back to constant if mismatch.
            return torch.ones((K, C, H, W), device=self.device)

        # DC-only amplitude
        A_dc = torch.zeros_like(A)
        A_dc[:, :, 0, 0] = A[:, :, 0, 0]

        # band-masked amplitude
        mask = self._band_mask(H, W, args.spec_band)  # [1,1,H,Wr]
        A_band = A * mask

        lam = float(args.spec_lambda)
        lam = max(0.0, min(1.0, lam))
        A_mix = (1.0 - lam) * A_dc + lam * A_band  # [1,C,H,Wr]

        # sample random phase
        # phase for all freqs except DC; DC kept real
        phase = torch.rand((K, C, H, Wr), device=self.device) * (2.0 * np.pi)
        # complex spectrum
        real = torch.cos(phase) * A_mix  # broadcast [1,C,H,Wr] -> [K,C,H,Wr]
        imag = torch.sin(phase) * A_mix
        real[:, :, 0, 0] = A_mix[:, :, 0, 0]  # enforce DC real
        imag[:, :, 0, 0] = 0.0

        X = torch.complex(real, imag)  # [K,C,H,Wr]
        x = torch.fft.irfft2(X, s=(H, W), dim=(-2, -1), norm='ortho')  # [K,C,H,W] real

        clamp_v = float(args.spec_clamp)
        if clamp_v > 0:
            x = x.clamp(min=-clamp_v, max=clamp_v)
        return x

    @torch.no_grad()
    def estimate_bias_logits(self, model: nn.Module, H: int, W: int) -> torch.Tensor:
        """
        Estimate bias logits by averaging model outputs over K random-phase probes.
        Returns: [num_class] (logits)
        """
        K = max(1, int(args.spec_k))
        probes = self.sample_probe(H, W, K)

        prev_mode = model.training
        if args.spec_mode == 'eval':
            model.eval()
        else:
            model.train()

        logits, _ = model(probes)  # [K,C]
        # restore
        model.train(prev_mode)

        return logits.mean(dim=0).detach()


# -------------------------
# Main
# -------------------------
def main():
    if not os.path.isdir(args.out):
        mkdir_p(args.out)

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

    # Scheme-3 state
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec_state = SpectralProbeState(device=device)

    # Resume
    title = 'fixcdmad-scheme3-' + args.dataset
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # restore amplitude stats if present
        if 'A_data' in checkpoint and checkpoint['A_data'] is not None:
            spec_state.A_data = checkpoint['A_data'].to(device)
            spec_state.inited = True
            spec_state.update_count = int(checkpoint.get('A_count', 0))
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        # stable 5 columns for your parsing scripts
        logger.set_names(['bACC', 'GM', 'bACC_debias', 'GM_debias', 'Top1'])

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f | SpecLambda: %.2f Band: %s' %
              (epoch + 1, args.epochs, state['lr'], float(args.spec_lambda), args.spec_band))

        train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, train_criterion, epoch, spec_state)

        test_acc1, testclassacc1, test_acc2, testclassacc2 = validate(test_loader, ema_model, criterion, mode='Test Stats ', epoch=epoch, spec_state=spec_state)

        GM = geometric_mean(testclassacc1)
        GM2 = geometric_mean(testclassacc2)

        print("without test debias bACC:", testclassacc1.mean(), "GM:", GM,
              "with test debias bACC:", testclassacc2.mean(), "GM:", GM2)

        logger.append([testclassacc1.mean(), GM, testclassacc2.mean(), GM2, test_acc1])

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # scheme-3 states
            'A_data': spec_state.A_data.detach().cpu() if spec_state.A_data is not None else None,
            'A_count': spec_state.update_count,
        }, epoch + 1)

    logger.close()


def geometric_mean(accperclass: np.ndarray) -> float:
    gm = 1.0
    for i in range(num_class):
        if accperclass[i] == 0:
            gm *= (1 / (100 * num_class)) ** (1 / num_class)
        else:
            gm *= (accperclass[i]) ** (1 / num_class)
    return float(gm)


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, spec_state: SpectralProbeState):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()

    # logging spectral stats
    spec_ready_m = AverageMeter()
    bias_norm_m = AverageMeter()

    end = time.time()
    bar = Bar('Training', max=args.val_iteration)

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()

    for batch_idx in range(args.val_iteration):
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

        # Update amplitude spectrum stats using labeled + weak unlabeled (both are in model input space)
        spec_state.update_from_batch(inputs_x)
        spec_state.update_from_batch(inputs_u)

        H, W = int(inputs_x.shape[2]), int(inputs_x.shape[3])
        ready = float(spec_state.inited and (spec_state.update_count >= int(args.spec_warmup_iters)))
        spec_ready_m.update(ready, 1)

        with torch.no_grad():
            outputs_u, _ = model(inputs_u)

            if epoch > args.debiasstart:
                biaseddegree = spec_state.estimate_bias_logits(model, H=H, W=W)
                bias_norm_m.update(biaseddegree.norm(p=2).item(), 1)
                outputs_u = outputs_u - biaseddegree.detach()

            targets_u2 = F.softmax(outputs_u, dim=1).detach()

        max_p, p_hat = torch.max(targets_u2, dim=1)
        p_hat = torch.zeros(int(args.unlabeledratio * batch_size), num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
        select_mask = max_p.ge(args.tau)
        select_mask = torch.cat([select_mask, select_mask], 0).float()

        # keep your original choice: soft targets
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
                 'Loss: {loss:.4f} | Lx: {loss_x:.4f} | Lu: {loss_u:.4f} | SpecReady: {sr:.2f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    sr=spec_ready_m.avg,
                )
        if epoch > args.debiasstart and bias_norm_m.count > 0:
            suffix += ' | ||b||: %.3f' % (bias_norm_m.avg,)
        bar.suffix = suffix
        bar.next()

    bar.finish()
    return (losses.avg, losses_x.avg, losses_u.avg)


def validate(valloader, model, criterion, mode, epoch: int, spec_state: SpectralProbeState):
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

    with torch.no_grad():
        # infer input size from first batch (for STL compatibility)
        first_batch = next(iter(valloader))
        x0 = first_batch[0].cuda()
        H, W = int(x0.shape[2]), int(x0.shape[3])

        # Estimate bias logits once for test-time debias (spectral probe family)
        biaseddegree = spec_state.estimate_bias_logits(model, H=H, W=W)

        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            outputs, _ = model(inputs)
            outputs2 = outputs - biaseddegree

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
                batch=batch_idx + 1, size=len(valloader), data=data_time.avg, bt=batch_time.avg,
                top1=top1.avg, top5=top5.avg
            )
            bar.next()
        bar.finish()

    # Normalize per-class accuracy denominators
    if args.dataset == 'cifar10':
        accperclass = accperclass / 1000
        accperclass2 = accperclass2 / 1000
    elif args.dataset == 'stl10':
        accperclass = accperclass / 800
        accperclass2 = accperclass2 / 800
    elif args.dataset == 'cifar100':
        accperclass = accperclass / 100
        accperclass2 = accperclass2 / 100

    return (top1.avg, accperclass, top1debias.avg, accperclass2)


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
