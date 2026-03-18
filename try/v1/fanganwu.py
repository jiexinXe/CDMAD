# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP)
# Modified (Scheme-5): "theoretically-grounded" optimal baseline picture via PRIOR-MATCHING optimization
#
# Core idea (Scheme-5, standalone):
#   We view logit debias as subtracting an additive bias vector b, which is realized as the logits of a baseline picture I_b:
#       b := z_theta(I_b)  (up to a constant, we keep centering optional)
#   We choose I_b by solving a principled objective:
#       Make the *marginal* predicted class distribution on weak unlabeled data, after debias, match a target prior π.
#
#   Concretely, for a minibatch U:
#       p_bar(b) = mean_u softmax( z_theta(u) - b )
#       Optimize baseline image I_b to minimize:
#           KL( p_bar(z(I_b)) || π ) + λ_tv TV(I_b) + λ_l2 ||I_b||^2
#
#   This yields a baseline picture that is not hand-crafted, and is directly tied to a statistical property (prior matching),
#   rather than a heuristic color/phase choice.
#
# Independence note:
#   This script is intentionally independent from Scheme-1/2/3/4:
#     - no mixture probe distributions (Scheme-1)
#     - no b0/Δb decomposition (Scheme-2)
#     - no spectral FFT probes (Scheme-3)
#     - no activation-stat matching (Scheme-4)

from __future__ import print_function

import argparse
import os
import shutil
import time
import random
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
parser = argparse.ArgumentParser(description='PyTorch FixMatch Training (CDMAD baseline, Scheme-5 prior-matched baseline picture)')

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
# Scheme-5: Prior-matched baseline picture
# -------------------------
def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Isotropic TV on a batch of images: x [B,C,H,W]."""
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
    return (dh.abs().mean() + dw.abs().mean())


class PriorMatchedBaseline:
    """
    Maintain one or multiple baseline pictures (nn.Parameter) and optimize them by prior matching:

        p_bar = mean_u softmax( z(u) - z(I_b) )
        L = KL(p_bar || π) + λ_tv TV(I_b) + λ_l2 ||I_b||^2

    The "bias vector" used for debias is b = mean_m z(I_bm).
    """
    def __init__(self, device: torch.device, num_class: int):
        self.device = device
        self.num_class = num_class
        self.probes = None  # nn.Parameter [M,3,H,W]
        self.opt = None
        self.HW = None
        self.global_iter = 0
        self.last_metrics = {}

        # target prior π set later when we know labeled counts
        self.pi = None  # [C]

    def set_target_prior(self, pi: np.ndarray):
        pi = np.asarray(pi, dtype=np.float32)
        pi = pi / max(1e-12, float(pi.sum()))
        self.pi = torch.tensor(pi, device=self.device).clamp(min=1e-8)

    def _ensure_shape(self, H: int, W: int):
        if self.probes is not None and self.HW == (H, W):
            return
        M = max(1, int(args.pm_num_probes))
        # init near 0 (often corresponds to dataset mean in normalized space)
        init = torch.zeros((M, 3, H, W), device=self.device)
        init += 0.05 * torch.randn_like(init)
        self.probes = nn.Parameter(init)
        self.opt = optim.Adam([self.probes], lr=float(args.pm_lr))
        self.HW = (H, W)

    @staticmethod
    def _center_logits(b: torch.Tensor) -> torch.Tensor:
        return b - b.mean()

    def _bias_from_probes(self, model: nn.Module) -> torch.Tensor:
        logits_b, _ = model(self.probes)  # [M,C]
        b = logits_b.mean(dim=0)          # [C]
        if args.pm_center:
            b = self._center_logits(b)
        return b

    def maybe_update(self, model: nn.Module, inputs_u_weak: torch.Tensor):
        """
        Update baseline probes periodically using weak unlabeled batch.
        - model params are NOT updated here.
        - baseline probes are updated with gradients.

        inputs_u_weak: [B,3,H,W] (already on GPU)
        """
        self.global_iter += 1
        B, C, H, W = inputs_u_weak.shape
        self._ensure_shape(H, W)

        if self.pi is None:
            # default uniform
            self.set_target_prior(np.ones(self.num_class, dtype=np.float32))

        # warmup + schedule
        if self.global_iter < int(args.pm_warmup_iters):
            return
        if int(args.pm_update_freq) > 1 and (self.global_iter % int(args.pm_update_freq) != 0):
            return

        # Freeze model params, optimize probes only
        prev_train = model.training
        if args.pm_mode == 'eval':
            model.eval()
        else:
            model.train()

        # compute logits_u once (no grad needed)
        with torch.no_grad():
            logits_u, _ = model(inputs_u_weak)  # [B,C]

        # optimize probes for a few steps
        kl_vals = []
        tv_vals = []
        l2_vals = []
        bnorm_vals = []

        for _ in range(max(1, int(args.pm_steps))):
            self.opt.zero_grad(set_to_none=True)

            # need grad through probes
            b = self._bias_from_probes(model)  # [C] with grad
            bnorm_vals.append(float(b.detach().norm(p=2).item()))

            debiased = logits_u - b.view(1, -1)  # [B,C]
            p_bar = F.softmax(debiased, dim=1).mean(dim=0).clamp(min=1e-8)  # [C]

            pi = self.pi
            if args.pm_kl == 'forward':
                # KL(p_bar || pi) = sum p_bar (log p_bar - log pi)
                kl = torch.sum(p_bar * (torch.log(p_bar) - torch.log(pi)))
            else:
                # KL(pi || p_bar) = sum pi (log pi - log p_bar)
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

        # restore model mode
        model.train(prev_train)

        self.last_metrics = {
            'kl': float(np.mean(kl_vals)) if len(kl_vals) else 0.0,
            'tv': float(np.mean(tv_vals)) if len(tv_vals) else 0.0,
            'l2': float(np.mean(l2_vals)) if len(l2_vals) else 0.0,
            'bnorm': float(np.mean(bnorm_vals)) if len(bnorm_vals) else 0.0,
        }

    @torch.no_grad()
    def bias_logits(self, model: nn.Module, H: int, W: int) -> torch.Tensor:
        """Return bias logits b = mean z(I_b)."""
        self._ensure_shape(H, W)
        prev_train = model.training
        model.eval()
        logits_b, _ = model(self.probes.detach())
        model.train(prev_train)
        b = logits_b.mean(dim=0)
        if args.pm_center:
            b = self._center_logits(b)
        return b.detach()


# -------------------------
# Main
# -------------------------
def main():
    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio, args.imbalancetype)
    U_SAMPLES_PER_CLASS = make_imb_data(args.num_max_u, num_class, args.imb_ratio_u, args.imbalancetype)

    # set weight decay heuristic
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

    # Model
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

    # define target prior π
    if args.pm_target == 'uniform':
        pi = np.ones(num_class, dtype=np.float32)
    else:
        # labeled empirical prior from imbalance counts
        pi = np.asarray(N_SAMPLES_PER_CLASS, dtype=np.float32)
    pmb.set_target_prior(pi)

    # Resume
    title = 'fixcdmad-scheme5-' + args.dataset
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # restore probes if present
        if 'pmb_probes' in checkpoint and checkpoint['pmb_probes'] is not None:
            # infer H,W from tensor
            probes = checkpoint['pmb_probes'].to(device)
            H, W = int(probes.shape[2]), int(probes.shape[3])
            pmb._ensure_shape(H, W)
            with torch.no_grad():
                pmb.probes.copy_(probes)
        if 'pmb_global_iter' in checkpoint:
            pmb.global_iter = int(checkpoint['pmb_global_iter'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        # stable 5 columns for your parsing scripts
        logger.set_names(['bACC', 'GM', 'bACC_debias', 'GM_debias', 'Top1'])

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f | PMTarget: %s | KL: %s' %
              (epoch + 1, args.epochs, state['lr'], args.pm_target, args.pm_kl))

        train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
              ema_optimizer, train_criterion, epoch, pmb)

        test_acc1, testclassacc1, test_acc2, testclassacc2 = validate(
            test_loader, ema_model, criterion, mode='Test Stats ', epoch=epoch, pmb=pmb
        )

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
            # scheme-5 states
            'pmb_probes': pmb.probes.detach().cpu() if pmb.probes is not None else None,
            'pmb_global_iter': int(pmb.global_iter),
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


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch: int, pmb: PriorMatchedBaseline):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()

    # pmb metrics
    kl_m = AverageMeter()
    tv_m = AverageMeter()
    bnorm_m = AverageMeter()

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

        H, W = int(inputs_x.shape[2]), int(inputs_x.shape[3])

        # --- Scheme-5: update baseline probes (uses weak unlabeled only) ---
        # This is independent from model updates (only probes are updated here).
        with torch.enable_grad():
            pmb.maybe_update(model, inputs_u)

        if pmb.last_metrics:
            kl_m.update(pmb.last_metrics.get('kl', 0.0), 1)
            tv_m.update(pmb.last_metrics.get('tv', 0.0), 1)
            bnorm_m.update(pmb.last_metrics.get('bnorm', 0.0), 1)

        with torch.no_grad():
            outputs_u, _ = model(inputs_u)

            if epoch > args.debiasstart:
                biaseddegree = pmb.bias_logits(model, H, W)
                outputs_u = outputs_u - biaseddegree.detach()

            targets_u2 = F.softmax(outputs_u, dim=1).detach()

        max_p, p_hat = torch.max(targets_u2, dim=1)
        p_hat = torch.zeros(int(args.unlabeledratio * batch_size), num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
        select_mask = max_p.ge(args.tau)
        select_mask = torch.cat([select_mask, select_mask], 0).float()

        # Keep your original choice: soft targets
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
    return (losses.avg, losses_x.avg, losses_u.avg)


def validate(valloader, model, criterion, mode: str, epoch: int, pmb: PriorMatchedBaseline):
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
        # infer H,W from first batch (for STL)
        first_batch = next(iter(valloader))
        x0 = first_batch[0].cuda()
        H, W = int(x0.shape[2]), int(x0.shape[3])

        if args.pm_test_debias:
            biaseddegree = pmb.bias_logits(model, H, W)
        else:
            biaseddegree = torch.zeros(num_class, device=x0.device)

        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

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
