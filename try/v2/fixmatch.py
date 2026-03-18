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
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from scipy import optimize
import csv
import contextlib

# ---------------- Diagnostics utilities ----------------
DIAG_FIELDS = [
    "epoch","iter","lr",
    "probe_mode","probe_types","probe_ref",
    "b_entropy","b_kl_uniform","b_l2","b_ema_l2_delta",
    "probe_jsd_avg_vs_ref",
    "accept_rate_raw","accept_rate_debias","flip_rate_raw_vs_debias",
    "pl_head_mass_raw","pl_tail_mass_raw","pl_head_tail_ratio_raw",
    "pl_head_mass_debias","pl_tail_mass_debias","pl_head_tail_ratio_debias",
    "bn_drift_l2","corr_logpb_logpi_labeled","corr_logpb_logpi_pl_raw"]
DIAG_F = None
DIAG_WRITER = None
PROBE_EMA = None  # torch tensor [C]

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

def bn_snapshot(model: nn.Module):
    stats = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and getattr(m, "track_running_stats", False):
            if m.running_mean is not None:
                stats.append(m.running_mean.detach().clone())
            if m.running_var is not None:
                stats.append(m.running_var.detach().clone())
    return stats

def bn_drift_l2(before, after) -> float:
    if not before or not after:
        return 0.0
    s = 0.0
    for a, b in zip(after, before):
        s += (a.float() - b.float()).pow(2).sum().item()
    return float(math.sqrt(s))

def _probe_parse_list(s: str):
    items = [x.strip().lower() for x in s.split(",") if x.strip()]
    return items if items else ["white"]

def make_probe_like(x_ref: torch.Tensor, kind: str) -> torch.Tensor:
    """Construct a probe image in the same input space as x_ref (already preprocessed).
    Supports both deterministic (const/mean/min/max) and stochastic (gauss/unif) probes.
    """
    kind = kind.lower()
    B, C, H, W = x_ref.shape
    ch_mean = x_ref.mean(dim=(0,2,3), keepdim=True)   # [1,C,1,1]
    ch_std  = x_ref.std(dim=(0,2,3), keepdim=True).clamp_min(1e-6)
    ch_min  = x_ref.amin(dim=(0,2,3), keepdim=True)
    ch_max  = x_ref.amax(dim=(0,2,3), keepdim=True)

    # ---------- Stochastic probes ----------
    if kind in ("gauss","gaussian","normal"):
        # Maximum entropy under (mean, var) constraints: Gaussian
        noise = torch.randn((1, C, H, W), device=x_ref.device, dtype=x_ref.dtype)
        return (noise * ch_std + ch_mean).contiguous()

    if kind in ("unif","uniform"):
        # Uniform with matched mean/std per-channel: std = (b-a)/sqrt(12)
        width = ch_std * math.sqrt(12.0)
        a = ch_mean - 0.5 * width
        b = ch_mean + 0.5 * width
        u = torch.rand((1, C, H, W), device=x_ref.device, dtype=x_ref.dtype)
        return (u * (b - a) + a).contiguous()

    # ---------- Deterministic / constant probes ----------
    if kind in ("const1","ones"):
        base = torch.ones_like(ch_mean)
    elif kind in ("const0","zeros"):
        base = torch.zeros_like(ch_mean)
    elif kind in ("const05","half"):
        base = 0.5 * torch.ones_like(ch_mean)
    elif kind == "mean":
        base = ch_mean
    elif kind == "zero":
        base = torch.zeros_like(ch_mean)
    elif kind == "white":
        base = ch_max
    elif kind == "black":
        base = ch_min
    elif kind == "gray":
        base = 0.5 * (ch_min + ch_max)
    elif kind in ("red","green","blue") and C >= 3:
        base = ch_min.clone()
        if kind == "red":
            base[:,0:1] = ch_max[:,0:1]
        elif kind == "green":
            base[:,1:2] = ch_max[:,1:2]
        else:
            base[:,2:3] = ch_max[:,2:3]
    else:
        base = ch_mean

    return base.expand(1, C, H, W).contiguous()

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
    # Return logits [C] for a single probe image.
    probe_mode = probe_mode.lower()
    if probe_mode == "eval":
        with _temporary_eval(model):
            with torch.inference_mode():
                logits, _ = model(x_probe)
    else:
        # train mode (may update BN running stats!)
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
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0015, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--num_max', type=int, default=1500,
                        help='Number of samples in the maximal class')
parser.add_argument('--num_max_u', type=int, default=3000,
                        help='Number of samples in the maximal class')
#parser.add_argument('--label_ratio', type=float, default=30, help='percentage of labeled data')
parser.add_argument('--imb_ratio', type=int, default=100, help='Imbalance ratio')
parser.add_argument('--imb_ratio_u', type=float, default=100, help='Imbalance ratio')
parser.add_argument('--step', action='store_true', help='Type of class-imbalance')
parser.add_argument('--val-iteration', type=int, default=500,
                        help='Frequency for the evaluation')

parser.add_argument('--tau', default=0, type=float, help='hyper-parameter for pseudo-label of FixMatch')
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--wd', default=0.04, type=float)

#dataset and imbalanced type
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset')
parser.add_argument('--imbalancetype', type=str, default='long', help='Long tailed or step imbalanced')
parser.add_argument('--unlabeledratio', type=float, default=2, help='Long tailed or step imbalanced')
parser.add_argument('--debiasstart', type=int, default=100, help='Long tailed or step imbalanced')
# ---------------- Diagnostics / Probe settings (baseline picture study) ----------------
parser.add_argument('--diag', dest='diag_enable', action='store_true',
                    help='Enable diagnostics logging for baseline picture / bias probe')
parser.add_argument('--no-diag', dest='diag_enable', action='store_false',
                    help='Disable diagnostics logging')
parser.set_defaults(diag_enable=True)

parser.add_argument('--diag-freq', type=int, default=50,
                    help='Log diagnostics every N iterations (default: 50)')
parser.add_argument('--diag-file', type=str, default='diag_metrics.csv',
                    help='Diagnostics CSV filename saved under --out')
parser.add_argument('--probe-mode', type=str, default='eval', choices=['eval', 'train'],
                    help='Compute probe logits in eval() (no BN update) or train() mode (will update BN). '
                         'Use eval for normal training; train is only for back-action experiments.')
parser.add_argument('--probe-types', type=str, default='white,black,gray,gauss,unif',
                    help='Comma-separated probe types to evaluate, e.g. "const1,const0,const05,white,black,gray,gauss,unif,mean,red,green,blue". '
                         'Probes are constructed in the same input space as the current batch.')
parser.add_argument('--probe-ref', type=str, default='white',
                    help='Which probe in --probe-types to use as the main debias reference (default: white)')
parser.add_argument('--probe-ema', type=float, default=0.99,
                    help='EMA factor for probe logits stability tracking (default: 0.99)')

parser.add_argument('--probe-samples', type=int, default=1,
                    help='Monte-Carlo samples for stochastic probes (gauss/unif). Use >1 to reduce estimator variance.')
parser.add_argument('--log-bn-drift', action='store_true',
                    help='If set, log BN running stats drift caused by probe forward (useful when --probe-mode=train)')


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
# np.random.seed(args.manualSeed)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio,args.imbalancetype)
    U_SAMPLES_PER_CLASS = make_imb_data(args.num_max_u, num_class, args.imb_ratio_u,args.imbalancetype)
    # ---- Diagnostics globals ----
    global CLASS_COUNTS_L, HEAD_MASK, TAIL_MASK
    CLASS_COUNTS_L = np.array(N_SAMPLES_PER_CLASS, dtype=np.int64)
    global LOG_PI_L
    pi_l = CLASS_COUNTS_L.astype(np.float64) / (CLASS_COUNTS_L.sum() + 1e-12)
    LOG_PI_L = torch.from_numpy(np.log(pi_l + 1e-12)).float()
    HEAD_MASK, TAIL_MASK = _head_tail_masks_from_counts(CLASS_COUNTS_L)

    # Init diagnostics CSV
    diag_init(args.out)

    if np.array(N_SAMPLES_PER_CLASS).sum()+np.array(U_SAMPLES_PER_CLASS).sum() >= 30000 or args.dataset == 'stl10':
        args.wd=0.01
    if args.dataset == 'cifar10':
        train_labeled_set, train_unlabeled_set,test_set = dataset.get_cifar10('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS,rand_number=args.manualSeed)
    elif args.dataset == 'stl10':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_stl10('./data', N_SAMPLES_PER_CLASS,args.out,rand_number=args.manualSeed)

    elif args.dataset =='cifar100':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_cifar100('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS,rand_number=args.manualSeed)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                          drop_last=True)
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

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in params) / 1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    # Resume
    title = 'fixcdmad-' + args.dataset
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['bACC_raw','GM_raw','bACC_debias','GM_debias','top1_raw','top1_debias'])
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))


        train(labeled_trainloader,unlabeled_trainloader,model, optimizer,ema_optimizer,train_criterion,epoch)

        test_acc1, testclassacc1, test_acc2, testclassacc2= validate(test_loader, ema_model,criterion,mode='Test Stats ')
        GM = 1
        for i in range(num_class):
            if testclassacc1[i] == 0:
                # To prevent the N/A values, we set the minimum value as 0.001
                GM *= (1 / (100 * num_class)) ** (1 / num_class)
            else:
                GM *= (testclassacc1[i]) ** (1 / num_class)
        GM2 = 1
        for i in range(num_class):
            if testclassacc2[i] == 0:
                # To prevent the N/A values, we set the minimum value as 0.001
                GM2 *= (1 / (100 * num_class)) ** (1 / num_class)
            else:
                GM2 *= (testclassacc2[i]) ** (1 / num_class)

        print("raw: top1:", test_acc1, "bACC:", testclassacc1.mean(), "GM:", GM, "| debias: top1:", test_acc2, "bACC:", testclassacc2.mean(), "GM:", GM2)
        logger.append([testclassacc1.mean(), GM, testclassacc2.mean(), GM2, test_acc1, test_acc2])

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),

                'optimizer' : optimizer.state_dict(),
            }, epoch + 1)

    logger.close()
def train(labeled_trainloader,unlabeled_trainloader, model,optimizer, ema_optimizer, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
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


        # ---------------- Probe / bias estimate (CDMAD-style) ----------------
        probe_kinds = _probe_parse_list(args.probe_types)
        probe_ref = args.probe_ref.lower() if args.probe_ref.lower() in probe_kinds else probe_kinds[0]

        # Optional BN drift measurement (only meaningful for probe_mode=train)
        bn_drift = 0.0
        bn_before = None
        if args.diag_enable and args.log_bn_drift and args.probe_mode.lower() == "train" and (batch_idx % args.diag_freq == 0):
            bn_before = bn_snapshot(model)

        with torch.no_grad():
            # 1) Compute probe logits for each probe kind (in the same input space as current batch)
            probe_logits = {}
            for k in probe_kinds:
                # Monte-Carlo averaging for stochastic probes (gauss/unif)
                n_samp = max(1, int(args.probe_samples)) if k in ("gauss","gaussian","normal","unif","uniform") else 1
                if n_samp == 1:
                    x_probe = make_probe_like(inputs_u, k)
                    probe_logits[k] = forward_probe_logits(model, x_probe, args.probe_mode)
                else:
                    logits_acc = []
                    for _ in range(n_samp):
                        x_probe = make_probe_like(inputs_u, k)
                        logits_acc.append(forward_probe_logits(model, x_probe, args.probe_mode))
                    probe_logits[k] = torch.stack(logits_acc, dim=0).mean(dim=0)


            # BN drift after probes
            if bn_before is not None:
                bn_after = bn_snapshot(model)
                bn_drift = bn_drift_l2(bn_before, bn_after)

            # Main bias vector b (logits) used for debias
            b_logits = probe_logits[probe_ref].detach()  # [C]

            # 2) Unlabeled weak logits (raw & debiased for diagnostics)
            outputs_u_raw, _ = model(inputs_u)
            outputs_u_debias = outputs_u_raw
            if epoch > args.debiasstart:
                outputs_u_debias = outputs_u_raw - b_logits.view(1, -1)

            targets_u_raw = F.softmax(outputs_u_raw, dim=1).detach()
            targets_u2 = F.softmax(outputs_u_debias, dim=1).detach()


        max_p, p_hat = torch.max(targets_u2, dim=1)
        p_hat = torch.zeros(int(args.unlabeledratio*batch_size), num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
        select_mask = max_p.ge(args.tau)

        select_mask = torch.cat([select_mask, select_mask], 0).float()

        # ---------------- Diagnostics logging (periodic) ----------------
        if args.diag_enable and (batch_idx % args.diag_freq == 0):
            global PROBE_EMA
            b_entropy = entropy_from_logits(b_logits)
            b_kl_u = kl_to_uniform_from_logits(b_logits)
            b_l2 = float(torch.norm(b_logits, p=2).item())

            # EMA stability of b
            if PROBE_EMA is None:
                PROBE_EMA = b_logits.detach().clone()
            else:
                PROBE_EMA = args.probe_ema * PROBE_EMA + (1.0 - args.probe_ema) * b_logits.detach()
            b_ema_delta = float(torch.norm((b_logits.detach() - PROBE_EMA), p=2).item())

            # Probe sensitivity: average JS divergence vs ref
            p_ref = F.softmax(probe_logits[probe_ref], dim=-1)
            js_vals = []
            for k in probe_kinds:
                if k == probe_ref:
                    continue
                pk = F.softmax(probe_logits[k], dim=-1)
                js_vals.append(js_divergence(p_ref, pk))
            js_avg = float(np.mean(js_vals)) if len(js_vals) > 0 else 0.0

            # Acceptance & flip diagnostics (raw vs debiased)
            max_p_raw, y_raw = torch.max(targets_u_raw, dim=1)
            max_p_deb, y_deb = torch.max(targets_u2, dim=1)
            accept_raw = float(max_p_raw.ge(args.tau).float().mean().item())
            accept_deb = float(max_p_deb.ge(args.tau).float().mean().item())
            flip_rate = float((y_raw != y_deb).float().mean().item())

            # Mass on head/tail classes
            counts_raw = torch.bincount(y_raw, minlength=num_class)
            counts_deb = torch.bincount(y_deb, minlength=num_class)
            h_raw, t_raw, r_raw = _mass_ratio_from_counts(counts_raw.cpu(), HEAD_MASK, TAIL_MASK)
            h_deb, t_deb, r_deb = _mass_ratio_from_counts(counts_deb.cpu(), HEAD_MASK, TAIL_MASK)

            lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else 0.0


            # Correlation: does probe estimate resemble (log) labeled prior? does it track pseudo-label drift?
            logpb = F.log_softmax(b_logits, dim=-1).detach().cpu()
            corr_logpb_l = pearson_corr(logpb, LOG_PI_L) if LOG_PI_L is not None else 0.0

            pl_counts = torch.bincount(y_raw, minlength=num_class).float()
            pl_pi = (pl_counts / pl_counts.sum().clamp_min(1.0)).clamp_min(1e-12)
            log_pi_pl = torch.log(pl_pi).cpu()
            corr_logpb_pl = pearson_corr(logpb, log_pi_pl)

            diag_write({
                "epoch": epoch,
                "iter": batch_idx,
                "lr": lr,
                "probe_mode": args.probe_mode,
                "probe_types": args.probe_types,
                "probe_ref": probe_ref,
                "b_entropy": b_entropy,
                "b_kl_uniform": b_kl_u,
                "b_l2": b_l2,
                "b_ema_l2_delta": b_ema_delta,
                "probe_jsd_avg_vs_ref": js_avg,
                "accept_rate_raw": accept_raw,
                "accept_rate_debias": accept_deb,
                "flip_rate_raw_vs_debias": flip_rate,
                "pl_head_mass_raw": h_raw,
                "pl_tail_mass_raw": t_raw,
                "pl_head_tail_ratio_raw": r_raw,
                "pl_head_mass_debias": h_deb,
                "pl_tail_mass_debias": t_deb,
                "pl_head_tail_ratio_debias": r_deb,
                "bn_drift_l2": bn_drift,
                "corr_logpb_logpi_labeled": corr_logpb_l,
                "corr_logpb_logpi_pl_raw": corr_logpb_pl
            })

        #all_targets = torch.cat([targets_x2, p_hat, p_hat], dim=0)
        #else:
        all_targets = torch.cat([targets_x2, targets_u2, targets_u2], dim=0)


        logits_x,_= model(inputs_x)
        logits_u2,_ = model(inputs_u2)
        logits_u3,_ = model(inputs_u3)

        logits_u = torch.cat([logits_u2,logits_u3],dim=0)

        Lx, Lu = criterion(logits_x,all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)

        loss=Lx+Lu
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg)

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


        # Build a probe aligned to validation input space using the first batch
        probe_kinds = _probe_parse_list(args.probe_types)
        probe_ref = args.probe_ref.lower() if args.probe_ref.lower() in probe_kinds else probe_kinds[0]

        first_batch = next(iter(valloader))
        x_ref = first_batch[0].cuda(non_blocking=True) if isinstance(first_batch, (list, tuple)) else first_batch.cuda(non_blocking=True)
        x_probe = make_probe_like(x_ref, probe_ref)

        biaseddegree = forward_probe_logits(model, x_probe, "eval").view(1, -1)
        for batch_idx, (inputs, targets, _) in enumerate(valloader):

            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            targetsonehot = torch.zeros(inputs.size()[0], num_class).scatter_(1, targets.cpu().view(-1, 1).long(), 1)
            outputs,_=model(inputs)
            outputs2=outputs-biaseddegree

            score = F.softmax(outputs)
            score2 = F.softmax(outputs2)


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


def f(x, a, b, c, d):
    return np.sum(a * b * np.exp(-1 * x/c)) - d


def make_imb_data(max_num, class_num, gamma,imb):
    if imb == 'long':
        mu = np.power(1/gamma, 1/(class_num - 1))
        class_num_list = []
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(int(max_num / gamma))
            else:
                class_num_list.append(int(max_num * np.power(mu, i)))
        print(class_num_list)
    if imb=='step':
        class_num_list = []
        for i in range(class_num):
            if i < int((class_num) / 2):
                class_num_list.append(int(max_num))
            else:
                class_num_list.append(int(max_num / gamma))
        print(class_num_list)
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

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

if __name__ == '__main__':
    try:
        main()
    finally:
        diag_close()
