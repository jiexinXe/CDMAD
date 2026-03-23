# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP)
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math
import numpy as np

# 恢复 np.int 为内置 int
setattr(np, 'int', int)

import wrn as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p
import csv
import contextlib

# ---------------- Diagnostics utilities ----------------
DIAG_FIELDS = [
    "epoch", "iter", "lr",
    "baseline_picture",
    "b_entropy", "b_kl_uniform", "b_l2",
    "accept_rate_raw", "accept_rate_debias", "flip_rate_raw_vs_debias",
    "pl_head_mass_raw", "pl_tail_mass_raw", "pl_head_tail_ratio_raw",
    "pl_head_mass_debias", "pl_tail_mass_debias", "pl_head_tail_ratio_debias",
    "corr_logpb_logpi_labeled", "corr_logpb_logpi_pl_raw",
    "stable_conflict_rate", "proto_active_rate",
    "zra_energy"
]

DIAG_F = None
DIAG_WRITER = None
EMA_MEAN_IMG = None  # torch tensor [1, C, H, W]

PROBE_MODE = "eval"
MEAN_EMA_DECAY = 0.999
PROTO_MOMENTUM = 0.99

# === ZRA-Base 全局变量 ===
X_BASE = None           # 可学习的零响应基准图 Parameter
OPT_BASE = None         # 基准图的优化器
# ========================

# === Labeled prototype EMA for weak semantic guidance ===
PROTO_L_EMA = None      # torch tensor [C, D]
PROTO_L_VALID = None    # torch bool tensor [C]
# ================================================

# These will be set in main()
CLASS_COUNTS_L = None  # np.ndarray [C]
HEAD_MASK = None       # torch.BoolTensor [C] on CPU
TAIL_MASK = None       # torch.BoolTensor [C] on CPU

LOG_PI_L = None       # torch.FloatTensor [C] on CPU (log labeled prior)
def _safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))

def entropy_from_logits(logits_1d: torch.Tensor) -> float:
    # Entropy in nats for logits [C]
    p = F.softmax(logits_1d, dim=-1)
    h = -(p * _safe_log(p)).sum()
    return float(h.item())

def kl_to_uniform_from_logits(logits_1d: torch.Tensor) -> float:
    # KL(softmax(logits) || Uniform)
    c = logits_1d.numel()
    h = entropy_from_logits(logits_1d)
    return float(math.log(c) - h)

def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    # Jensen-Shannon divergence for prob vectors p,q
    m = 0.5 * (p + q)
    kl_pm = (p * (_safe_log(p) - _safe_log(m))).sum()
    kl_qm = (q * (_safe_log(q) - _safe_log(m))).sum()
    js = 0.5 * (kl_pm + kl_qm)
    return float(js.item())

def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    # Pearson correlation between 1D tensors
    a = a.float().view(-1)
    b = b.float().view(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (torch.norm(a, p=2) * torch.norm(b, p=2)).clamp_min(1e-12)
    return float((a * b).sum().item() / denom.item())


@torch.no_grad()
def update_ema_mean_image(x_ref: torch.Tensor, alpha: float):
    global EMA_MEAN_IMG
    batch_mean_img = x_ref.mean(dim=0, keepdim=True)
    if EMA_MEAN_IMG is None:
        EMA_MEAN_IMG = batch_mean_img.detach().clone()
    else:
        if EMA_MEAN_IMG.shape != batch_mean_img.shape:
            EMA_MEAN_IMG = batch_mean_img.detach().clone()
        else:
            EMA_MEAN_IMG.mul_(alpha).add_(batch_mean_img * (1.0 - alpha))

@torch.no_grad()
def get_single_baseline_probe(x_ref: torch.Tensor, mode: str) -> torch.Tensor:
    mode = mode.lower()
    if mode in ("white", "black", "gray", "mean"):
        return make_probe_like(x_ref, mode)
    if mode == "batch_mean":
        return x_ref.mean(dim=0, keepdim=True).contiguous()
    return make_probe_like(x_ref, "white")

def make_probe_like(x_ref: torch.Tensor, kind: str) -> torch.Tensor:
    kind = kind.lower()
    _, C, H, W = x_ref.shape
    ch_mean = x_ref.mean(dim=(0, 2, 3), keepdim=True)
    ch_min = x_ref.amin(dim=(0, 2, 3), keepdim=True)
    ch_max = x_ref.amax(dim=(0, 2, 3), keepdim=True)

    if kind == "mean":
        base = ch_mean
    elif kind == "white":
        base = ch_max
    elif kind == "black":
        base = ch_min
    elif kind == "gray":
        base = 0.5 * (ch_min + ch_max)
    else:
        base = ch_mean

    return base.expand(1, C, H, W).contiguous()

@torch.no_grad()
def get_baseline_probe(x_ref: torch.Tensor, baseline_picture: str) -> torch.Tensor:
    mode = baseline_picture.lower()
    if mode == "batch_mean":
        return x_ref.mean(dim=0, keepdim=True).contiguous()
    return make_probe_like(x_ref, mode)


def _get_centered_prior_dir(device, dtype) -> torch.Tensor:
    r = LOG_PI_L.to(device=device, dtype=dtype).clone()
    r = r - r.mean()
    r = r / (r.norm(p=2) + 1e-12)
    return r


def _project_logits_to_prior(logits_1d: torch.Tensor) -> torch.Tensor:
    """Project centered logits onto the labeled-prior direction."""
    r_hat = _get_centered_prior_dir(logits_1d.device, logits_1d.dtype)
    zc = logits_1d - logits_1d.mean()
    coeff = torch.dot(zc, r_hat)
    return coeff * r_hat


def _build_ppma_baseline(model: nn.Module, x_ref: torch.Tensor):
    """PPMA: Prior-Projected Mean Anchor.

    Start from the EMA mean image (stable, low-semantic anchor), then take one
    gradient step along the direction that most increases the prior-aligned
    centered-logit score. Finally, keep only the prior-projected bias component
    for debiasing.
    """
    if EMA_MEAN_IMG is not None and EMA_MEAN_IMG.shape == x_ref[:1].shape:
        x_anchor = EMA_MEAN_IMG.detach().clone().to(device=x_ref.device, dtype=x_ref.dtype)
    else:
        x_anchor = x_ref.mean(dim=0, keepdim=True).detach()

    ch_min = x_ref.amin(dim=(0, 2, 3), keepdim=True)
    ch_max = x_ref.amax(dim=(0, 2, 3), keepdim=True)
    r_hat = _get_centered_prior_dir(x_ref.device, x_ref.dtype)

    _req_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        x_var = x_anchor.clone().detach().requires_grad_(True)
        with _temporary_eval(model):
            logits_ppma, feat_ppma = _forward_logits_and_penultimate(model, x_var, class_dim=num_class)
            logits_1d = logits_ppma.squeeze(0)
            zc = logits_1d - logits_1d.mean()
            prior_score = torch.dot(zc, r_hat)
            feat_energy = (feat_ppma ** 2).mean()
            grad = torch.autograd.grad(prior_score, x_var, retain_graph=False, create_graph=False)[0]
    finally:
        for p, _rf in zip(model.parameters(), _req_flags):
            p.requires_grad_(_rf)

    anchor_std = x_anchor.detach().std().clamp_min(1e-6)
    step_radius = args.ppma_rho * anchor_std * math.sqrt(float(x_anchor.numel()))
    delta = step_radius * grad / (grad.norm(p=2) + 1e-12)
    x_ppma = x_anchor + delta
    x_ppma = torch.max(torch.min(x_ppma, ch_max), ch_min)

    with torch.no_grad():
        raw_b = forward_probe_logits(model, x_ppma.detach(), PROBE_MODE).detach()
        b_proj = _project_logits_to_prior(raw_b)

    return x_ppma.detach(), b_proj, float(feat_energy.item()), float(prior_score.item())


def _get_classifier_module(model: nn.Module, class_dim: int = None) -> nn.Module:
    """Best-effort retrieval of the final classifier Linear module [C, D].

    Prefer a linear layer whose out_features matches class_dim (e.g. 10/100)
    to avoid accidentally selecting an auxiliary projector head.
    """
    preferred_names = ["fc", "linear", "classifier", "head"]
    for name in preferred_names:
        if hasattr(model, name):
            layer = getattr(model, name)
            if isinstance(layer, nn.Linear):
                if class_dim is None or layer.weight.shape[0] == class_dim:
                    return layer

    candidates = []
    for mod_name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if class_dim is None or m.weight.shape[0] == class_dim:
                candidates.append((mod_name, m))

    if candidates:
        # choose the last matching linear layer in module traversal order
        return candidates[-1][1]

    raise RuntimeError(f"Could not locate classifier Linear module for USC with class_dim={class_dim}.")


def _get_classifier_weight(model: nn.Module, class_dim: int = None) -> torch.Tensor:
    return _get_classifier_module(model, class_dim=class_dim).weight


def _forward_logits_and_penultimate(model: nn.Module, x: torch.Tensor, class_dim: int):
    """Run model forward and capture the true penultimate feature, i.e. the input
    to the final classifier layer. This is robust even when model(x) returns an
    auxiliary tensor instead of penultimate features.
    """
    classifier = _get_classifier_module(model, class_dim=class_dim)
    captured = {}

    def _pre_hook(module, inputs):
        feat = inputs[0]
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        captured['feat'] = feat

    handle = classifier.register_forward_pre_hook(_pre_hook)
    try:
        out = model(x)
    finally:
        handle.remove()

    logits = out[0] if isinstance(out, (tuple, list)) else out
    feat = captured.get('feat', None)
    if feat is None:
        raise RuntimeError("USC failed to capture penultimate feature from classifier input.")
    if feat.dim() > 2:
        feat = feat.view(feat.size(0), -1)
    return logits, feat


def _project_orthogonal(feat: torch.Tensor, v_bias: torch.Tensor) -> torch.Tensor:
    """Project features to the orthogonal complement of v_bias."""
    if feat.dim() == 1:
        proj = torch.dot(feat, v_bias) * v_bias
        return feat - proj
    proj = (feat * v_bias.unsqueeze(0)).sum(dim=1, keepdim=True) * v_bias.unsqueeze(0)
    return feat - proj


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


@torch.no_grad()
def _update_labeled_prototypes(feat_x_teacher: torch.Tensor, targets_x: torch.Tensor, momentum: float):
    global PROTO_L_EMA, PROTO_L_VALID
    if feat_x_teacher is None or targets_x is None:
        return
    feat_x_teacher = feat_x_teacher.detach()
    targets_x = targets_x.detach().long()
    C, D = num_class, feat_x_teacher.size(1)
    device = feat_x_teacher.device
    if PROTO_L_EMA is None or PROTO_L_EMA.shape != (C, D):
        PROTO_L_EMA = torch.zeros(C, D, device=device, dtype=feat_x_teacher.dtype)
        PROTO_L_VALID = torch.zeros(C, device=device, dtype=torch.bool)
    else:
        PROTO_L_EMA = PROTO_L_EMA.to(device=device, dtype=feat_x_teacher.dtype)
        PROTO_L_VALID = PROTO_L_VALID.to(device=device)

    for c in targets_x.unique(sorted=False):
        c = int(c.item())
        mask = targets_x.eq(c)
        if mask.any():
            feat_mean = feat_x_teacher[mask].mean(dim=0)
            if not PROTO_L_VALID[c]:
                PROTO_L_EMA[c] = feat_mean
                PROTO_L_VALID[c] = True
            else:
                PROTO_L_EMA[c].mul_(momentum).add_(feat_mean * (1.0 - momentum))


def _gather_projected_labeled_prototypes(labels: torch.Tensor, v_bias: torch.Tensor, ref_feat: torch.Tensor):
    global PROTO_L_EMA, PROTO_L_VALID
    if PROTO_L_EMA is None or PROTO_L_VALID is None:
        return None, None
    labels = labels.long()
    proto = PROTO_L_EMA.to(device=ref_feat.device, dtype=ref_feat.dtype)[labels]
    proto_valid = PROTO_L_VALID.to(device=ref_feat.device)[labels].float()
    proto_perp = _project_orthogonal(proto, v_bias.detach())
    return proto_perp, proto_valid

@contextlib.contextmanager
def _temporary_eval(model: nn.Module):
    was_training = model.training
    try:
        model.eval()
        yield
    finally:
        if was_training:
            model.train()

def forward_probe_logits(model: nn.Module, x_probe: torch.Tensor, probe_mode: str) -> torch.Tensor:
    probe_mode = probe_mode.lower()
    if probe_mode == "eval":
        with _temporary_eval(model):
            with torch.inference_mode():
                logits, _ = model(x_probe)
    else:
        with torch.inference_mode():
            logits, _ = model(x_probe)
    return logits.squeeze(0)

def diag_init(out_dir: str):
    global DIAG_F, DIAG_WRITER
    if not args.diag_enable:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, args.diag_file)
    is_new = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    DIAG_F = open(path, "a", newline="")
    DIAG_WRITER = csv.DictWriter(DIAG_F, fieldnames=DIAG_FIELDS)
    if is_new:
        DIAG_WRITER.writeheader()
        DIAG_F.flush()

def diag_close():
    global DIAG_F, DIAG_WRITER
    try:
        if DIAG_F is not None:
            DIAG_F.flush()
            DIAG_F.close()
    finally:
        DIAG_F = None
        DIAG_WRITER = None

def diag_write(row: dict):
    if (not args.diag_enable) or (DIAG_WRITER is None):
        return
    for k in DIAG_FIELDS:
        row.setdefault(k, "")
    DIAG_WRITER.writerow(row)
    DIAG_F.flush()

def _head_tail_masks_from_counts(counts_np: np.ndarray):
    c = len(counts_np)
    k = max(1, int(round(0.3 * c)))
    idx = np.argsort(-counts_np)  # descending
    head_idx = idx[:k]
    tail_idx = idx[-k:]
    head_mask = torch.zeros(c, dtype=torch.bool)
    tail_mask = torch.zeros(c, dtype=torch.bool)
    head_mask[torch.tensor(head_idx, dtype=torch.long)] = True
    tail_mask[torch.tensor(tail_idx, dtype=torch.long)] = True
    return head_mask, tail_mask

def _mass_ratio_from_counts(counts: torch.Tensor, head_mask: torch.Tensor, tail_mask: torch.Tensor):
    total = counts.sum().float().clamp_min(1.0)
    head = counts[head_mask].sum().float() / total
    tail = counts[tail_mask].sum().float() / total
    ratio = (head + 1e-12) / (tail + 1e-12)
    return float(head.item()), float(tail.item()), float(ratio.item())

parser = argparse.ArgumentParser(description='PyTorch fixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0015, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result', help='Directory to output the result')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num_max', type=int, default=1500, help='Number of samples in the maximal class')
parser.add_argument('--num_max_u', type=int, default=3000, help='Number of samples in the maximal class')
parser.add_argument('--imb_ratio', type=int, default=100, help='Imbalance ratio')
parser.add_argument('--imb_ratio_u', type=float, default=100, help='Imbalance ratio')
parser.add_argument('--val-iteration', type=int, default=500, help='Frequency for the evaluation')
parser.add_argument('--tau', default=0, type=float, help='hyper-parameter for pseudo-label of FixMatch')
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--wd', default=0.04, type=float)

parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset')
parser.add_argument('--imbalancetype', type=str, default='long', help='Long tailed or step imbalanced')
parser.add_argument('--unlabeledratio', type=float, default=2, help='Long tailed or step imbalanced')
parser.add_argument('--debiasstart', type=int, default=100, help='Long tailed or step imbalanced')

# ---------------- Diagnostics / ZRA / USC settings ----------------
parser.add_argument('--diag', dest='diag_enable', action='store_true', help='Enable diagnostics logging')
parser.add_argument('--no-diag', dest='diag_enable', action='store_false', help='Disable diagnostics logging')
parser.set_defaults(diag_enable=True)
parser.add_argument('--diag-freq', type=int, default=50, help='Log diagnostics every N iterations (default: 50)')
parser.add_argument('--diag-file', type=str, default='diag_metrics.csv', help='Diagnostics CSV filename saved under --out')

parser.add_argument('--baseline-picture', type=str, default='zra_base',
                choices=['white', 'black', 'gray', 'mean', 'batch_mean', 'zra_base', 'ppma'],
                help='Single baseline picture for debiasing. zra_base = original learned minimum-response anchor; ppma = Prior-Projected Mean Anchor.')
parser.add_argument('--fma-lr', type=float, default=0.01, help='Learning rate for ZRA-Base inner-loop optimization.')
parser.add_argument('--fma-tv-weight', type=float, default=0.0001, help='TV loss weight to keep ZRA-Base smooth.')
parser.add_argument('--ppma-rho', type=float, default=0.10, help='Relative per-pixel step size for PPMA local prior-revealing correction.')

parser.add_argument('--usc', dest='usc_enable', action='store_true', help='Enable Unbiased Subspace Consistency (USC).')
parser.add_argument('--no-usc', dest='usc_enable', action='store_false', help='Disable Unbiased Subspace Consistency (USC).')
parser.set_defaults(usc_enable=True)
parser.add_argument('--usc-start', type=int, default=-1, help='Epoch to start USC; -1 uses debiasstart+30.')
parser.add_argument('--usc-weight', type=float, default=0.75, help='Weight for USC loss.')
parser.add_argument('--usc-conflict-boost', type=float, default=0.5, help='Extra multiplicative weight for raw/debias conflict samples in USC.')

parser.add_argument('--proto-pull', dest='proto_pull_enable', action='store_true', help='Enable weak positive prototype pull in orthogonal subspace.')
parser.add_argument('--no-proto-pull', dest='proto_pull_enable', action='store_false', help='Disable weak positive prototype pull.')
parser.set_defaults(proto_pull_enable=True)
parser.add_argument('--proto-weight', type=float, default=0.25, help='Weight for prototype pull loss.')
parser.add_argument('--proto-stable-base-weight', type=float, default=0.25, help='Base per-sample weight for stable selected samples in prototype pull; conflict samples are upweighted to 1.0.')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
if args.dataset=='cifar10':
    import dataset.fix_cifar10 as dataset
    print(f'==> Preparing imbalanced CIFAR10')
    num_class = 10
elif args.dataset=='cifar100':
    import dataset.fix_cifar100 as dataset
    print(f'==> Preparing imbalanced CIFAR100')
    num_class = 100
    args.wd=0.08
elif args.dataset=='stl10':
    import dataset.fix_stl10 as dataset
    print(f'==> Preparing imbalanced STL_10')
    num_class = 10
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio,args.imbalancetype)
    U_SAMPLES_PER_CLASS = make_imb_data(args.num_max_u, num_class, args.imb_ratio_u,args.imbalancetype)
    
    global CLASS_COUNTS_L, HEAD_MASK, TAIL_MASK
    CLASS_COUNTS_L = np.array(N_SAMPLES_PER_CLASS, dtype=np.int64)
    global LOG_PI_L
    pi_l = CLASS_COUNTS_L.astype(np.float64) / (CLASS_COUNTS_L.sum() + 1e-12)
    LOG_PI_L = torch.from_numpy(np.log(pi_l + 1e-12)).float()
    HEAD_MASK, TAIL_MASK = _head_tail_masks_from_counts(CLASS_COUNTS_L)

    diag_init(args.out)
    global PROTO_L_EMA, PROTO_L_VALID
    PROTO_L_EMA = None
    PROTO_L_VALID = None

    if np.array(N_SAMPLES_PER_CLASS).sum()+np.array(U_SAMPLES_PER_CLASS).sum() >= 30000 or args.dataset == 'stl10':
        args.wd=0.01
    if args.dataset == 'cifar10':
        train_labeled_set, train_unlabeled_set,test_set = dataset.get_cifar10('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS,rand_number=args.manualSeed)
    elif args.dataset == 'stl10':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_stl10('./data', N_SAMPLES_PER_CLASS,args.out,rand_number=args.manualSeed)
    elif args.dataset =='cifar100':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_cifar100('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS,rand_number=args.manualSeed)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=int(args.unlabeledratio*args.batch_size), shuffle=True, num_workers=4,drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=200, shuffle=False, num_workers=4)

    # Model
    print("==> creating WRN-28-2")

    def create_model(ema=False):
        model = models.WRN(2,num_classes=num_class)
        model = model.cuda()
        params = list(model.parameters())
        if ema:
            for param in params:
                param.detach_()
        return model, params

    model, params = create_model()
    ema_model,  _ = create_model(ema=True)

    print('    Total params: %.2fM' % (sum(p.numel() for p in params) / 1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    title = 'fixcdmad-' + args.dataset
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # [修改点] 恢复 ZRA-Base 的图
        if 'X_base' in checkpoint and checkpoint['X_base'] is not None:
            global X_BASE, OPT_BASE
            X_BASE = nn.Parameter(checkpoint['X_base'].cuda())
            OPT_BASE = optim.Adam([X_BASE], lr=args.fma_lr)
            
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['TrainLoss','LossX','LossU','LossUSC','LossProto','AcceptDebias','bL2','bEntropy','bACC_raw','GM_raw','bACC_debias','GM_debias','top1_raw','top1_debias'])
        
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_stats = train(labeled_trainloader, unlabeled_trainloader, model, ema_model, optimizer, ema_optimizer, train_criterion, epoch)

        test_acc1, testclassacc1, test_acc2, testclassacc2= validate(test_loader, ema_model,criterion,mode='Test Stats ')
        GM = 1
        for i in range(num_class):
            if testclassacc1[i] == 0:
                GM *= (1 / (100 * num_class)) ** (1 / num_class)
            else:
                GM *= (testclassacc1[i]) ** (1 / num_class)
        GM2 = 1
        for i in range(num_class):
            if testclassacc2[i] == 0:
                GM2 *= (1 / (100 * num_class)) ** (1 / num_class)
            else:
                GM2 *= (testclassacc2[i]) ** (1 / num_class)

        print("raw: top1:", test_acc1, "bACC:", testclassacc1.mean(), "GM:", GM, "| debias: top1:", test_acc2, "bACC:", testclassacc2.mean(), "GM:", GM2)
        logger.append([
            train_stats['loss'], train_stats['loss_x'], train_stats['loss_u'], train_stats['loss_usc'], train_stats['loss_proto'],
            train_stats['accept_debias'], train_stats['b_l2'], train_stats['b_entropy'],
            testclassacc1.mean(), GM, testclassacc2.mean(), GM2, test_acc1, test_acc2
        ])

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'X_base': X_BASE.data if X_BASE is not None else None, # 存档
            }, epoch + 1)

    logger.close()

def train(labeled_trainloader, unlabeled_trainloader, model, ema_model, optimizer, ema_optimizer, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_usc = AverageMeter()
    losses_proto = AverageMeter()
    accept_deb_meter = AverageMeter()
    b_l2_meter = AverageMeter()
    b_entropy_meter = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()

    for batch_idx in range(args.val_iteration):

        try:
            inputs_x, targets_x, _ = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x,  targets_x, _ = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = next(unlabeled_train_iter)

        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        targets_x2 = torch.zeros(batch_size, num_class).scatter_(1, targets_x.view(-1,1), 1)

        inputs_x,targets_x2 = inputs_x.cuda(),targets_x2.cuda(non_blocking=True)
        inputs_u, inputs_u2, inputs_u3  = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()


        # ---------------- ZRA-Base & Baseline Estimator ----------------
        zra_energy_now = 0.0
        global X_BASE, OPT_BASE
        
        with torch.no_grad():
            outputs_u_raw, features_u = model(inputs_u)
            update_ema_mean_image(inputs_u, MEAN_EMA_DECAY)

        if args.baseline_picture == 'zra_base':
            # 内循环能量最小化优化
            if X_BASE is None:
                # 依然以均值图像为最安全的初始化起点
                init_mean = EMA_MEAN_IMG.detach().clone() if EMA_MEAN_IMG is not None else inputs_u.mean(dim=0, keepdim=True).detach()
                X_BASE = nn.Parameter(init_mean)
                OPT_BASE = optim.Adam([X_BASE], lr=args.fma_lr)

            # Freeze model params so only X_BASE receives gradients (avoid unnecessary grad/instability)
            _req_flags = [p.requires_grad for p in model.parameters()]
            for p in model.parameters():
                p.requires_grad_(False)
            try:
                with _temporary_eval(model):
                    logits_base, features_base = model(X_BASE)

                    # 【核心】：不向任何中心对齐，直接最小化特征的激活能量（L2 Norm平方）
                    loss_energy = (features_base ** 2).mean()

                    # TV loss（用 mean 而不是 sum，避免对分辨率/尺寸敏感）
                    tv_loss = (X_BASE[:, :, 1:, :] - X_BASE[:, :, :-1, :]).abs().mean() + \
                              (X_BASE[:, :, :, 1:] - X_BASE[:, :, :, :-1]).abs().mean()

                    loss_base = loss_energy + args.fma_tv_weight * tv_loss

                    OPT_BASE.zero_grad(set_to_none=True)
                    loss_base.backward()
                    OPT_BASE.step()
            finally:
                for p, _rf in zip(model.parameters(), _req_flags):
                    p.requires_grad_(_rf)

            with torch.no_grad():
                # 防止优化出现极端亮斑，规范回自然图片的数值区间
                ch_min = inputs_u.amin(dim=(0,2,3), keepdim=True)
                ch_max = inputs_u.amax(dim=(0,2,3), keepdim=True)
                X_BASE.data = torch.max(torch.min(X_BASE.data, ch_max), ch_min)
                
                b_logits = forward_probe_logits(model, X_BASE, PROBE_MODE).detach()
                zra_energy_now = float(loss_energy.item())

        elif args.baseline_picture == 'ppma':
            X_BASE_ppma, b_logits, zra_energy_now, _ppma_score = _build_ppma_baseline(ema_model, inputs_u)
            X_BASE = X_BASE_ppma.detach().clone()

        else:
            # 兼容：测试 mean、white、black 走这里
            with torch.no_grad():

                baseline_probe = get_baseline_probe(inputs_u, args.baseline_picture)
                b_logits = forward_probe_logits(model, baseline_probe, PROBE_MODE).detach()


        usc_start_epoch = (args.debiasstart + 30) if args.usc_start < 0 else args.usc_start
        proto_start_epoch = max(args.debiasstart + 100, usc_start_epoch + 50)
        usc_enabled = args.usc_enable and (epoch > usc_start_epoch)
        proto_enabled = args.proto_pull_enable and (epoch > proto_start_epoch)
        feat_u_teacher = None
        feat_x_teacher = None
        v_bias = None
        if usc_enabled or proto_enabled:
            with torch.no_grad():
                with _temporary_eval(ema_model):
                    _, feat_u_teacher = _forward_logits_and_penultimate(ema_model, inputs_u, class_dim=num_class)
                    if proto_enabled:
                        _, feat_x_teacher = _forward_logits_and_penultimate(ema_model, inputs_x, class_dim=num_class)
                    if args.baseline_picture in ('zra_base', 'ppma') and X_BASE is not None:
                        raw_probe_logits_ema = forward_probe_logits(ema_model, X_BASE.detach(), PROBE_MODE).detach()
                        b_logits_ema = _project_logits_to_prior(raw_probe_logits_ema) if args.baseline_picture == 'ppma' else raw_probe_logits_ema
                    else:
                        baseline_probe_ema = get_baseline_probe(inputs_u, args.baseline_picture)
                        b_logits_ema = forward_probe_logits(ema_model, baseline_probe_ema, PROBE_MODE).detach()
                b_c_ema = b_logits_ema - b_logits_ema.mean()
                W_ema = _get_classifier_weight(ema_model, class_dim=b_c_ema.numel()).detach()
                v_bias = torch.mv(W_ema.t(), b_c_ema)
                v_bias = v_bias / (v_bias.norm(p=2) + 1e-12)

        with torch.no_grad():
            outputs_u_debias = outputs_u_raw
            if epoch > args.debiasstart:
                outputs_u_debias = outputs_u_raw - b_logits.view(1, -1)

            targets_u_raw = F.softmax(outputs_u_raw, dim=1).detach()
            targets_u2 = F.softmax(outputs_u_debias, dim=1).detach()

        max_p, p_hat = torch.max(targets_u2, dim=1)
        p_hat = torch.zeros(int(args.unlabeledratio*batch_size), num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
        select_mask_weak = max_p.ge(args.tau).float()
        select_mask = torch.cat([select_mask_weak, select_mask_weak], 0)

        max_p_raw, y_raw = torch.max(targets_u_raw, dim=1)
        max_p_deb, y_deb = torch.max(targets_u2, dim=1)
        conflict_mask = y_raw.ne(y_deb).float()
        usc_weight_weak = select_mask_weak.clone()
        if usc_enabled:
            usc_weight_weak = usc_weight_weak * (1.0 + args.usc_conflict_boost * conflict_mask)
        else:
            usc_weight_weak = torch.zeros_like(select_mask_weak)

        accept_raw = float(max_p_raw.ge(args.tau).float().mean().item())
        accept_deb = float(max_p_deb.ge(args.tau).float().mean().item())

        b_l2_now = float(torch.norm(b_logits, p=2).item())
        b_entropy_now = entropy_from_logits(b_logits)
        accept_deb_meter.update(accept_deb, 1)
        b_l2_meter.update(b_l2_now, 1)
        b_entropy_meter.update(b_entropy_now, 1)

        # ---------------- Diagnostics statistics (periodic) ----------------
        need_diag = args.diag_enable and (batch_idx % args.diag_freq == 0)
        if need_diag:
            b_entropy = b_entropy_now
            b_kl_u = kl_to_uniform_from_logits(b_logits)
            b_l2 = b_l2_now
            flip_rate = float((y_raw != y_deb).float().mean().item())

            counts_raw = torch.bincount(y_raw, minlength=num_class)
            counts_deb = torch.bincount(y_deb, minlength=num_class)
            h_raw, t_raw, r_raw = _mass_ratio_from_counts(counts_raw.cpu(), HEAD_MASK, TAIL_MASK)
            h_deb, t_deb, r_deb = _mass_ratio_from_counts(counts_deb.cpu(), HEAD_MASK, TAIL_MASK)

            lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else 0.0

            logpb = F.log_softmax(b_logits, dim=-1).detach().cpu()
            corr_logpb_l = pearson_corr(logpb, LOG_PI_L) if LOG_PI_L is not None else 0.0

            pl_counts = torch.bincount(y_raw, minlength=num_class).float()
            pl_pi = (pl_counts / pl_counts.sum().clamp_min(1.0)).clamp_min(1e-12)
            log_pi_pl = torch.log(pl_pi).cpu()
            corr_logpb_pl = pearson_corr(logpb, log_pi_pl)

        all_targets = torch.cat([targets_x2, targets_u2, targets_u2], dim=0)

        # 提取用于 Loss 计算的有标签、无标签特征
        logits_x, features_x = model(inputs_x)
        logits_u2, feat_u2 = _forward_logits_and_penultimate(model, inputs_u2, class_dim=num_class)
        logits_u3, feat_u3 = _forward_logits_and_penultimate(model, inputs_u3, class_dim=num_class)

        logits_u = torch.cat([logits_u2,logits_u3],dim=0)

        Lx, Lu = criterion(logits_x,all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)

        stable_conflict_rate = 0.0
        proto_active_rate = 0.0
        if usc_enabled and feat_u_teacher is not None and v_bias is not None:
            feat_teacher_perp = _project_orthogonal(feat_u_teacher.detach(), v_bias.detach())
            feat_u2_perp = _project_orthogonal(feat_u2, v_bias.detach())
            feat_u3_perp = _project_orthogonal(feat_u3, v_bias.detach())

            usc_cos_2 = 1.0 - F.cosine_similarity(feat_teacher_perp, feat_u2_perp, dim=1, eps=1e-8)
            usc_cos_3 = 1.0 - F.cosine_similarity(feat_teacher_perp, feat_u3_perp, dim=1, eps=1e-8)
            Lusc = 0.5 * (_masked_mean(usc_cos_2, usc_weight_weak) + _masked_mean(usc_cos_3, usc_weight_weak))
            if usc_weight_weak.sum().item() <= 0:
                Lusc = logits_x.new_tensor(0.0)
        else:
            feat_teacher_perp = None
            feat_u2_perp = None
            feat_u3_perp = None
            Lusc = logits_x.new_tensor(0.0)

        # Weak positive prototype pull in the same orthogonal subspace.
        # To keep it stable, prototypes are built only from labeled teacher features,
        # and the pull is applied only to selected + conflict + stable samples.
        if usc_enabled and feat_u_teacher is not None and v_bias is not None:
            with torch.no_grad():
                logits_u2_deb = logits_u2.detach() - b_logits.view(1, -1)
                logits_u3_deb = logits_u3.detach() - b_logits.view(1, -1)
                y_deb_u2 = logits_u2_deb.argmax(dim=1)
                y_deb_u3 = logits_u3_deb.argmax(dim=1)
                stable_mask = y_deb_u2.eq(y_deb) & y_deb_u3.eq(y_deb)
                stable_selected_mask = select_mask_weak.bool() & stable_mask
                stable_conflict_mask = stable_selected_mask & conflict_mask.bool()
                stable_conflict_rate = float(stable_conflict_mask.float().mean().item())
        else:
            stable_mask = None
            stable_selected_mask = None
            stable_conflict_mask = None

        if proto_enabled and feat_x_teacher is not None:
            proto_perp, proto_valid = _gather_projected_labeled_prototypes(y_deb, v_bias, feat_u2)
            if proto_perp is not None and stable_selected_mask is not None:
                base_w = float(args.proto_stable_base_weight)
                per_sample_proto_w = base_w + (1.0 - base_w) * conflict_mask
                proto_mask = stable_selected_mask.float() * proto_valid * per_sample_proto_w
                proto_active_rate = float((proto_mask > 0).float().mean().item())
                proto_cos_2 = 1.0 - F.cosine_similarity(feat_u2_perp, proto_perp.detach(), dim=1, eps=1e-8)
                proto_cos_3 = 1.0 - F.cosine_similarity(feat_u3_perp, proto_perp.detach(), dim=1, eps=1e-8)
                Lproto = 0.5 * (_masked_mean(proto_cos_2, proto_mask) + _masked_mean(proto_cos_3, proto_mask))
                if proto_mask.sum().item() <= 0:
                    Lproto = logits_x.new_tensor(0.0)
            else:
                Lproto = logits_x.new_tensor(0.0)
        else:
            Lproto = logits_x.new_tensor(0.0)

        if need_diag:
            diag_write({
                "epoch": epoch,
                "iter": batch_idx,
                "lr": lr,
                "baseline_picture": args.baseline_picture,
                "b_entropy": b_entropy,
                "b_kl_uniform": b_kl_u,
                "b_l2": b_l2,                "accept_rate_raw": accept_raw,
                "accept_rate_debias": accept_deb,
                "flip_rate_raw_vs_debias": flip_rate,
                "pl_head_mass_raw": h_raw,
                "pl_tail_mass_raw": t_raw,
                "pl_head_tail_ratio_raw": r_raw,
                "pl_head_mass_debias": h_deb,
                "pl_tail_mass_debias": t_deb,
                "pl_head_tail_ratio_debias": r_deb,                "corr_logpb_logpi_labeled": corr_logpb_l,
                "corr_logpb_logpi_pl_raw": corr_logpb_pl,
                "stable_conflict_rate": stable_conflict_rate,
                "proto_active_rate": proto_active_rate,
                "zra_energy": zra_energy_now
            })

        loss = Lx + Lu + args.usc_weight * Lusc + args.proto_weight * Lproto
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        losses_usc.update(Lusc.item(), inputs_x.size(0))
        losses_proto.update(Lproto.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        if proto_enabled and feat_x_teacher is not None:
            _update_labeled_prototypes(feat_x_teacher, targets_x.to(feat_x_teacher.device, non_blocking=True), PROTO_MOMENTUM)

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | Loss_usc: {loss_usc:.4f} | Loss_proto: {loss_proto:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    loss_usc=losses_usc.avg,
                    loss_proto=losses_proto.avg,
                    )
        bar.next()
    bar.finish()

    return {
        'loss': losses.avg,
        'loss_x': losses_x.avg,
        'loss_u': losses_u.avg,
        'loss_usc': losses_usc.avg,
        'loss_proto': losses_proto.avg,
        'accept_debias': accept_deb_meter.avg,
        'b_l2': b_l2_meter.avg,
        'b_entropy': b_entropy_meter.avg,
    }

def validate(valloader,model,criterion,mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1debias = AverageMeter()
    top5debias= AverageMeter()

    # switch to evaluate mode
    model.eval()

    accperclass = np.zeros((num_class))
    accperclass2 = np.zeros((num_class))

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    with torch.no_grad():
        first_batch = next(iter(valloader))
        x_ref = first_batch[0].cuda(non_blocking=True) if isinstance(first_batch, (list, tuple)) else first_batch.cuda(non_blocking=True)
        
        # 验证时提取 probe logits。PPMA only subtracts the prior-projected component.
        if args.baseline_picture in ('zra_base', 'ppma') and X_BASE is not None:
            raw_probe_logits = forward_probe_logits(model, X_BASE, PROBE_MODE)
            if args.baseline_picture == 'ppma':
                raw_probe_logits = _project_logits_to_prior(raw_probe_logits)
            biaseddegree = raw_probe_logits.view(1, -1)
        else:
            baseline_probe = get_baseline_probe(x_ref, args.baseline_picture)
            biaseddegree = forward_probe_logits(model, baseline_probe, PROBE_MODE).view(1, -1)

        for batch_idx, (inputs, targets, _) in enumerate(valloader):

            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            targetsonehot = torch.zeros(inputs.size()[0], num_class).scatter_(1, targets.cpu().view(-1, 1).long(), 1)
            outputs,_=model(inputs)
            outputs2=outputs-biaseddegree

            score = F.softmax(outputs, dim=1)
            score2 = F.softmax(outputs2, dim=1)

            prediction=torch.argmax(score,dim=1)
            prediction2 = torch.argmax(score2, dim=1)

            outputs2onehot = torch.zeros(inputs.size()[0], num_class).scatter_(1, prediction.cpu().view(-1, 1).long(), 1)
            outputs2onehot2 = torch.zeros(inputs.size()[0], num_class).scatter_(1, prediction2.cpu().view(-1, 1).long(), 1)

            accperclass = accperclass + torch.sum(targetsonehot * outputs2onehot, dim=0).cpu().detach().numpy().astype(np.int64)
            accperclass2 = accperclass2 + torch.sum(targetsonehot * outputs2onehot2, dim=0).cpu().detach().numpy().astype(
                np.int64)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            prec1debias, prec5debias = accuracy(outputs2, targets, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            top1debias.update(prec1debias.item(), inputs.size(0))
            top5debias.update(prec5debias.item(), inputs.size(0))

             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                          'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    if args.dataset=='cifar10':
        accperclass=accperclass/1000
        accperclass2 = accperclass2 / 1000
    elif args.dataset=='stl10':
        accperclass=accperclass/800
        accperclass2 = accperclass2 / 800
    elif args.dataset=='cifar100':
        accperclass=accperclass/100
        accperclass2 = accperclass2 / 100
    return (top1.avg, accperclass, top1debias.avg, accperclass2)



def make_imb_data(max_num, class_num, gamma,imb):
    if imb == 'long':
        mu = np.power(1/gamma, 1/(class_num - 1))
        class_num_list = []
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(int(max_num / gamma))
            else:
                class_num_list.append(int(max_num * np.power(mu, i)))
    if imb=='step':
        class_num_list = []
        for i in range(class_num):
            if i < int((class_num) / 2):
                class_num_list.append(int(max_num))
            else:
                class_num_list.append(int(max_num / gamma))
    return list(class_num_list)

def save_checkpoint(state, epoch, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if epoch % 100 == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_' + str(epoch) + '.pth.tar'))


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask):
        Lx = -torch.mean(torch.sum(torch.log(F.softmax(outputs_x, dim=1)+1e-8) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(torch.log(F.softmax(outputs_u, dim=1)+1e-8) * targets_u, dim=1) * mask)
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
            ema_param=ema_param.float()
            param=param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

if __name__ == '__main__':
    try:
        main()
    finally:
        diag_close()