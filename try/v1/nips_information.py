# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP)
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math
import csv
from pathlib import Path

import numpy as np

# Restore np.int for legacy code
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
# Info-Debias (MaxEnt baseline + teacher-only correction) + Debug CSV
#
# Key changes (stability-oriented):
#  1) Baseline pictures are sampled from a maximum-entropy moment-matched family:
#        x_p = mu + s * sigma * eps   (mu/sigma are EMA stats of the training stream)
#     (no longer optimizing baseline via model outputs, which can collapse to a one-class probe)
#  2) Debias is applied ONLY to the EMA teacher distribution for pseudo-label targets.
#     Student logits (and supervised logits_x) are NOT shifted, so the model does not learn in a shifted-logit space.
#  3) Pseudo-label acceptance mask is computed from RAW teacher confidence (pre-debias) to keep accept-rate stable.
#  4) Keep CSV logging for diagnosis.
# -------------------------

# -------------------------
# MaxEnt Moment-Matched Baseline (simple, stable)
#   - Baseline samples are drawn from a maximum-entropy distribution that matches low-level moments of the
#     (augmented, normalized) training stream:  x_p = mu + s * sigma * eps.
#   - This avoids the degeneracy we observed when directly optimizing the baseline image using model outputs
#     (which can collapse to an "adversarial one-class probe").
# -------------------------

_BASELINE_SAMPLES = 16
_BASELINE_STAT_MOMENTUM = 0.99
_BASELINE_NOISE_SCALE = 0.50   # fixed scale relative to per-pixel sigma
_BASELINE_BLUR_K = 3           # fixed low-pass to suppress high-frequency structure

_BIAS_EMA_MOMENTUM = 0.90
_BIAS_CLIP_RANGE = 6.0         # clip (max-min) of bias logits to stabilize debias

_BETA_RAMP_EPOCHS = 50         # fixed ramp length after debiasstart

def _beta_ramp(epoch: int, debiasstart: int) -> float:
    """Fixed piecewise-linear ramp for beta (no new hyperparameters exposed)."""
    if epoch <= debiasstart:
        return 0.0
    t = epoch - debiasstart
    return float(min(1.0, t / float(_BETA_RAMP_EPOCHS)))


def _safe_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(probs * (probs + eps).log()).sum(dim=-1)


def _b_stats(b: torch.Tensor) -> dict:
    """b: (1,K) centered logit bias."""
    b1 = b.view(-1)
    p = torch.softmax(b1, dim=0)
    ent = float(_safe_entropy(p).item())
    return {
        'b_l2': float(torch.norm(b1, p=2).item()),
        'b_range': float((b1.max() - b1.min()).item()),
        'b_entropy': ent,
        'b_maxprob': float(p.max().item()),
        'b_argmax': int(torch.argmax(p).item()),
    }


def _group_acc(acc_per_class: np.ndarray, class_counts: list) -> dict:
    """Return many/medium/few accuracy (simple thresholds, common in LTSSL)."""
    counts = np.asarray(class_counts)
    many = counts >= 100
    medium = (counts < 100) & (counts >= 20)
    few = counts < 20

    def _mean(mask):
        if mask.sum() == 0:
            return float('nan')
        return float(acc_per_class[mask].mean())

    return {
        'acc_many': _mean(many),
        'acc_medium': _mean(medium),
        'acc_few': _mean(few),
        'n_many': int(many.sum()),
        'n_medium': int(medium.sum()),
        'n_few': int(few.sum()),
    }


class InfoBaseline(torch.nn.Module):
    """
    MaxEnt Moment-Matched Baseline (MEMB).

    We keep the class name `InfoBaseline` to minimize changes elsewhere, but the mechanism is different:
      - Maintain EMA estimates of per-pixel mean (mu) and std (sigma) of the training stream.
      - Sample baseline pictures from the maximum-entropy (diagonal Gaussian) family under these moment constraints:
            x_p = mu + noise_scale * sigma * eps
      - Estimate bias logits b as the EMA teacher's mean logits on these baseline samples.

    This avoids optimizing the baseline using model outputs (which can collapse to a one-class "adversarial probe"),
    and keeps the method simple & stable.
    """
    def __init__(self):
        super().__init__()
        self.mu = None        # (1,C,H,W) EMA mean image
        self.sigma = None     # (1,C,H,W) EMA std image
        self.b_ema = None     # (1,K) EMA bias logits

        self.inited = False

    @property
    def is_initialized(self) -> bool:
        # Backward-compatible alias used elsewhere in the training loop
        return bool(self.inited)

        self.last_step = None  # dict for debug csv

    @torch.no_grad()
    def maybe_init_from(self, like: torch.Tensor):
        if self.inited:
            return
        mu = like.mean(dim=0, keepdim=True)
        sigma = like.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        self.mu = mu.detach().clone()
        self.sigma = sigma.detach().clone()
        self.inited = True

    @torch.no_grad()
    def update_data_stats(self, x: torch.Tensor):
        """Update mu/sigma using EMA over the *training stream* (already augmented & normalized)."""
        self.maybe_init_from(x)
        m = x.mean(dim=0, keepdim=True)
        s = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)

        mom = _BASELINE_STAT_MOMENTUM
        self.mu.mul_(mom).add_(m * (1.0 - mom))
        self.sigma.mul_(mom).add_(s * (1.0 - mom))

        # debug fields expected by the existing CSV header
        self.last_step = {
            'baseline_mi': float('nan'),
            'baseline_stat': float('nan'),
            'baseline_loss': float('nan'),
            'mu_mean': self.mu.mean().item(),
            'mu_std': self.mu.std(unbiased=False).item(),
            'mu_min': self.mu.min().item(),
            'mu_max': self.mu.max().item(),
        }

    @torch.no_grad()
    def _sample(self, like: torch.Tensor, n: int = _BASELINE_SAMPLES) -> torch.Tensor:
        """Sample baseline pictures; clamp to the value range of `like` to avoid OOD extremes."""
        self.maybe_init_from(like)
        device = like.device
        eps = torch.randn((n,) + tuple(self.mu.shape[1:]), device=device)
        x = self.mu.to(device) + (_BASELINE_NOISE_SCALE * self.sigma.to(device)) * eps

        # Low-pass filter to suppress high-frequency structure (fixed, no hyper-parameter search)
        if _BASELINE_BLUR_K and _BASELINE_BLUR_K > 1:
            k = _BASELINE_BLUR_K
            x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

        # Clamp to observed range (per-batch global range is enough here)
        lo = like.min().item()
        hi = like.max().item()
        x = x.clamp(min=lo, max=hi)
        return x

    @torch.no_grad()
    def bias_logit(self, model: nn.Module, like: torch.Tensor, num_class: int) -> torch.Tensor:
        """Return (1,K) debias logits b (mean-centered), with EMA smoothing and range clipping."""
        # sample baseline pictures
        xb = self._sample(like=like, n=_BASELINE_SAMPLES)

        # forward
        model_was_training = model.training
        model.eval()
        logits_b, _ = model(xb)
        if model_was_training:
            model.train()

        b_hat = logits_b.mean(dim=0, keepdim=True)  # (1,K)
        b_hat = b_hat - b_hat.mean(dim=1, keepdim=True)  # remove class-constant shift

        # init / ema
        if (self.b_ema is None) or (self.b_ema.shape != b_hat.shape):
            self.b_ema = b_hat.detach().clone()
        else:
            mom = _BIAS_EMA_MOMENTUM
            self.b_ema.mul_(mom).add_(b_hat * (1.0 - mom))

        b = self.b_ema

        # Stabilize magnitude: clip range (max-min) to a fixed constant
        b_range = (b.max() - b.min()).detach().clamp_min(1e-6)
        if b_range.item() > _BIAS_CLIP_RANGE:
            b = b * (_BIAS_CLIP_RANGE / b_range)

        return b.detach()

parser = argparse.ArgumentParser(description='PyTorch fixMatch Training')
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.0015, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--out', default='result')
parser.add_argument('--manualSeed', type=int, default=0)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--num_max', type=int, default=1500)
parser.add_argument('--num_max_u', type=int, default=3000)
parser.add_argument('--imb_ratio', type=int, default=100)
parser.add_argument('--imb_ratio_u', type=float, default=100)
parser.add_argument('--step', action='store_true')
parser.add_argument('--val-iteration', type=int, default=500)
parser.add_argument('--tau', default=0, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--wd', default=0.04, type=float)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--imbalancetype', type=str, default='long')
parser.add_argument('--unlabeledratio', type=float, default=2)
parser.add_argument('--debiasstart', type=int, default=100)

args = parser.parse_args()

# Global learned baseline
BASELINE = InfoBaseline()

state = {k: v for k, v in args._get_kwargs()}

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

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


def validate(valloader, model, num_class_local, class_counts, mode='Test'):
    top1 = AverageMeter()
    top1debias = AverageMeter()
    model.eval()

    accperclass = np.zeros((num_class_local))
    accperclass2 = np.zeros((num_class_local))

    with torch.no_grad():
        b = None
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            targetsonehot = torch.zeros(inputs.size(0), num_class_local).scatter_(1, targets.cpu().view(-1, 1).long(), 1)

            outputs, _ = model(inputs)
            if b is None:
                BASELINE.maybe_init_from(inputs)
                b = BASELINE.bias_logit(model, like=inputs, num_class=num_class_local)
            outputs2 = outputs - b

            prediction = torch.argmax(outputs, dim=1)
            prediction2 = torch.argmax(outputs2, dim=1)
            outputs_onehot = torch.zeros(inputs.size(0), num_class_local).scatter_(1, prediction.cpu().view(-1, 1).long(), 1)
            outputs_onehot2 = torch.zeros(inputs.size(0), num_class_local).scatter_(1, prediction2.cpu().view(-1, 1).long(), 1)

            accperclass += torch.sum(targetsonehot * outputs_onehot, dim=0).cpu().numpy().astype(np.int64)
            accperclass2 += torch.sum(targetsonehot * outputs_onehot2, dim=0).cpu().numpy().astype(np.int64)

            prec1, _ = accuracy(outputs, targets, topk=(1, 5))
            prec1d, _ = accuracy(outputs2, targets, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top1debias.update(prec1d.item(), inputs.size(0))

    # Normalize per dataset
    if args.dataset == 'cifar10':
        denom = 1000
    elif args.dataset == 'stl10':
        denom = 800
    elif args.dataset == 'cifar100':
        denom = 100
    else:
        denom = 1

    accperclass = accperclass / denom
    accperclass2 = accperclass2 / denom

    return top1.avg, accperclass, top1debias.avg, accperclass2


def train(labeled_trainloader, unlabeled_trainloader, model, ema_model, optimizer, ema_optimizer, criterion, epoch, num_class_local):
    model.train()
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    beta = _beta_ramp(epoch, args.debiasstart)

    # pseudo-label stats
    pl_hist = torch.zeros(num_class_local, device='cpu')
    pl_mean_conf = 0.0
    pl_accept_rate = 0.0
    n_pl_batches = 0

    # Baseline update window (starts earlier, but still applied gradually via beta)
    bar = Bar('Training', max=args.val_iteration)

    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, _ = next(labeled_train_iter)
        except Exception:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2, inputs_u3), _, _ = next(unlabeled_train_iter)
        except Exception:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), _, _ = next(unlabeled_train_iter)

        batch_size = inputs_x.size(0)
        targets_x2 = torch.zeros(batch_size, num_class_local).scatter_(1, targets_x.view(-1, 1), 1)

        inputs_x = inputs_x.cuda()
        targets_x2 = targets_x2.cuda(non_blocking=True)
        inputs_u = inputs_u.cuda()
        inputs_u2 = inputs_u2.cuda()
        inputs_u3 = inputs_u3.cuda()

        # baseline init + stats
        BASELINE.maybe_init_from(inputs_x)
        BASELINE.update_data_stats(torch.cat([inputs_x, inputs_u], dim=0))

        with torch.no_grad():
            b = BASELINE.bias_logit(ema_model, like=inputs_x, num_class=num_class_local)  # (1,K)

            # Teacher pseudo-labels from EMA model:
            #   - use RAW confidence for mask to keep acceptance stable
            #   - use debiased distribution only for targets (so student does not learn in a shifted-logit space)
            out_u_t_raw, _ = ema_model(inputs_u)

            p_raw = F.softmax(out_u_t_raw, dim=1)
            max_p, _ = torch.max(p_raw, dim=1)
            select_mask = max_p.ge(args.tau)

            out_u_t = out_u_t_raw
            if beta > 0:
                out_u_t = out_u_t - beta * b
            targets_u2 = F.softmax(out_u_t, dim=1).detach()
        # record pseudo-label stats
        pl_mean_conf += float(max_p.mean().item())
        pl_accept_rate += float(select_mask.float().mean().item())
        pl_pred = torch.argmax(targets_u2, dim=1).cpu()
        pl_hist += torch.bincount(pl_pred, minlength=num_class_local).float()
        n_pl_batches += 1

        select_mask = torch.cat([select_mask, select_mask], 0).float()
        all_targets = torch.cat([targets_x2, targets_u2, targets_u2], dim=0)

        # Student forward
        logits_x, _ = model(inputs_x)
        logits_u2, _ = model(inputs_u2)
        logits_u3, _ = model(inputs_u3)
        # NOTE: do NOT shift student logits; only teacher targets are debiased.

        logits_u = torch.cat([logits_u2, logits_u3], dim=0)
        Lx, Lu = criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)
        loss = Lx + Lu

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        bar.suffix = f'({batch_idx+1}/{args.val_iteration}) beta:{beta:.3f} | Lx:{Lx.item():.4f} Lu:{Lu.item():.4f}'
        bar.next()

    bar.finish()

    # normalize pseudo-label stats
    if n_pl_batches > 0:
        pl_mean_conf /= n_pl_batches
        pl_accept_rate /= n_pl_batches
    pl_hist_sum = float(pl_hist.sum().item())
    pl_hist = (pl_hist / pl_hist_sum).numpy() if pl_hist_sum > 0 else np.zeros((num_class_local,))

    # b stats
    with torch.no_grad():
        b_stats = _b_stats(b)

    return {
        'beta': beta,
        'pl_mean_conf': pl_mean_conf,
        'pl_accept_rate': pl_accept_rate,
        'pl_hist': pl_hist,
        **b_stats,
        **(BASELINE.last_step or {}),
    }


def _ensure_csv(path: Path, header: list):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def main():
    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio, args.imbalancetype)
    U_SAMPLES_PER_CLASS = make_imb_data(args.num_max_u, num_class, args.imb_ratio_u, args.imbalancetype)

    if np.array(N_SAMPLES_PER_CLASS).sum() + np.array(U_SAMPLES_PER_CLASS).sum() >= 30000 or args.dataset == 'stl10':
        args.wd = 0.01

    if args.dataset == 'cifar10':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_cifar10('./data', N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS, rand_number=args.manualSeed)
    elif args.dataset == 'stl10':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_stl10('./data', N_SAMPLES_PER_CLASS, args.out, rand_number=args.manualSeed)
    elif args.dataset == 'cifar100':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_cifar100('./data', N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS, rand_number=args.manualSeed)
    else:
        raise ValueError

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=int(args.unlabeledratio * args.batch_size), shuffle=True, num_workers=4, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=200, shuffle=False, num_workers=4)

    print('==> creating WRN-28-2')

    def create_model(ema=False):
        m = models.WRN(2, num_classes=num_class).cuda()
        params = list(m.parameters())
        if ema:
            for p in params:
                p.detach_()
        return m, params

    model, params = create_model()
    ema_model, _ = create_model(ema=True)
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in params) / 1e6))

    train_criterion = SemiLoss()
    optimizer = optim.Adam(params, lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)

    # logger (kept for backward compatibility)
    title = 'fixcdmad-' + args.dataset
    logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
    logger.set_names(['bACC(no debias)', 'GM(no debias)', 'bACC(with debias)', 'GM(with debias)', 'Top1(no debias)', 'Top1(with debias)'])

    # debug csv
    metrics_csv = Path(args.out) / 'debug_metrics.csv'
    metrics_header = [
        'epoch', 'beta',
        'top1_no', 'top1_with', 'bacc_no', 'bacc_with', 'gm_no', 'gm_with',
        'acc_many_no', 'acc_medium_no', 'acc_few_no',
        'acc_many_with', 'acc_medium_with', 'acc_few_with',
        'pl_mean_conf', 'pl_accept_rate',
        'b_l2', 'b_range', 'b_entropy', 'b_maxprob', 'b_argmax',
        'baseline_mi', 'baseline_stat', 'baseline_loss',
        'mu_mean', 'mu_std', 'mu_min', 'mu_max',
    ]
    _ensure_csv(metrics_csv, metrics_header)

    # pseudo-label histogram csv
    pl_csv = Path(args.out) / 'pseudolabel_hist.csv'
    pl_header = ['epoch'] + [f'pl_frac_c{i}' for i in range(num_class)]
    _ensure_csv(pl_csv, pl_header)

    # bias vector csv (only for small K)
    bias_csv = None
    if num_class <= 20:
        bias_csv = Path(args.out) / 'bias_vector.csv'
        bias_header = ['epoch'] + [f'b{i}' for i in range(num_class)]
        _ensure_csv(bias_csv, bias_header)

    best_with = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        print(f'\nEpoch: [{epoch + 1} | {args.epochs}] LR: {state["lr"]:.6f}')

        train_stats = train(labeled_trainloader, unlabeled_trainloader, model, ema_model, optimizer, ema_optimizer, train_criterion, epoch, num_class)

        top1_no, acc_no, top1_with, acc_with = validate(test_loader, ema_model, num_class, N_SAMPLES_PER_CLASS)

        # GM
        gm_no = float(np.prod(np.clip(acc_no, 1 / (100 * num_class), None)) ** (1 / num_class))
        gm_with = float(np.prod(np.clip(acc_with, 1 / (100 * num_class), None)) ** (1 / num_class))

        bacc_no = float(acc_no.mean())
        bacc_with = float(acc_with.mean())

        # group acc
        g_no = _group_acc(acc_no, N_SAMPLES_PER_CLASS)
        g_with = _group_acc(acc_with, N_SAMPLES_PER_CLASS)

        print('without test debias bACC:', bacc_no, 'GM:', gm_no, 'with test debias bACC:', bacc_with, 'GM', gm_with)

        logger.append([bacc_no, gm_no, bacc_with, gm_with, top1_no, top1_with])

        # write debug csv
        with metrics_csv.open('a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, train_stats.get('beta', 0.0),
                top1_no, top1_with, bacc_no, bacc_with, gm_no, gm_with,
                g_no['acc_many'], g_no['acc_medium'], g_no['acc_few'],
                g_with['acc_many'], g_with['acc_medium'], g_with['acc_few'],
                train_stats.get('pl_mean_conf', float('nan')),
                train_stats.get('pl_accept_rate', float('nan')),
                train_stats.get('b_l2', float('nan')),
                train_stats.get('b_range', float('nan')),
                train_stats.get('b_entropy', float('nan')),
                train_stats.get('b_maxprob', float('nan')),
                train_stats.get('b_argmax', -1),
                train_stats.get('baseline_mi', float('nan')),
                train_stats.get('baseline_stat', float('nan')),
                train_stats.get('baseline_loss', float('nan')),
                train_stats.get('mu_mean', float('nan')),
                train_stats.get('mu_std', float('nan')),
                train_stats.get('mu_min', float('nan')),
                train_stats.get('mu_max', float('nan')),
            ])

        # pseudo-label hist
        with pl_csv.open('a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1] + [float(x) for x in train_stats.get('pl_hist', np.zeros((num_class,)))])

        # bias vector (small K)
        if bias_csv is not None:
            with torch.no_grad():
                # use current b_ema (already centered)
                # compute from a tiny probe based on current test batch distribution is ok here
                # we just want a consistent trend; validate already computed b internally.
                # so we recompute with a safe dummy tensor if mu exists.
                if BASELINE.is_initialized:
                    # create a small dummy tensor from mu itself
                    dummy = BASELINE.mu.detach().clone()
                    bvec = BASELINE.bias_logit(ema_model, like=dummy, num_class=num_class).view(-1).cpu().numpy()
                    with bias_csv.open('a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1] + [float(v) for v in bvec.tolist()])

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch + 1)

    logger.close()


if __name__ == '__main__':
    main()