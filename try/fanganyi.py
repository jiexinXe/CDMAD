# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP)
# Modified (Scheme-1): probe distribution + low-variance bias estimator (baseline picture as a probing distribution)

from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np

# Restore np.int to built-in int (compat)
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
parser = argparse.ArgumentParser(description='PyTorch fixMatch Training (CDMAD baseline, Scheme-1 probe distribution)')

# Optimization options
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0015, type=float, metavar='LR', help='initial learning rate')

# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result', help='Directory to output the result')

# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')

# Device options
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

# Long-tail setup
parser.add_argument('--num_max', type=int, default=1500, help='Number of samples in the maximal labeled class')
parser.add_argument('--num_max_u', type=int, default=3000, help='Number of samples in the maximal unlabeled class')
parser.add_argument('--imb_ratio', type=int, default=100, help='Imbalance ratio (labeled)')
parser.add_argument('--imb_ratio_u', type=float, default=100, help='Imbalance ratio (unlabeled)')
parser.add_argument('--step', action='store_true', help='Type of class-imbalance (legacy)')

parser.add_argument('--val-iteration', type=int, default=500, help='Iterations per epoch (train)')

# FixMatch hyper-params
parser.add_argument('--tau', default=0, type=float, help='threshold for pseudo-label in FixMatch')
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--wd', default=0.04, type=float)

# Dataset / imbalance type
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset: cifar10/cifar100/stl10')
parser.add_argument('--imbalancetype', type=str, default='long', help='long or step imbalance')
parser.add_argument('--unlabeledratio', type=float, default=2, help='unlabeled batch multiplier')
parser.add_argument('--debiasstart', type=int, default=100, help='epoch to start debias (warm start)')

# -------------------------
# Scheme-1: Probe distribution & bias estimator
# -------------------------
parser.add_argument('--debias-ramp', type=int, default=20,
                    help='linear ramp epochs after debiasstart (0 -> no ramp, immediately full debias)')
parser.add_argument('--probe-k', type=int, default=16,
                    help='number of probe samples to estimate bias each update')
parser.add_argument('--probe-update-freq', type=int, default=1,
                    help='update bias every N iterations (only when debias weight > 0)')
parser.add_argument('--probe-mode', type=str, default='eval', choices=['eval', 'train'],
                    help='use model.eval() or model.train() when forwarding probe images (eval is recommended to avoid BN pollution)')
parser.add_argument('--probe-ema', type=float, default=0.99,
                    help='EMA decay for the estimated bias vector')
parser.add_argument('--data-stat-ema', type=float, default=0.99,
                    help='EMA decay for running data mean/std used by probe sampling')

# Probe mixture (distribution) in *input tensor space* (after transforms)
parser.add_argument('--probe-mix-const', type=float, default=0.40,
                    help='probability of constant probe = running data mean')
parser.add_argument('--probe-mix-jitter', type=float, default=0.30,
                    help='probability of jittered-mean probe (mean + gaussian)')
# remaining probability goes to low-frequency probes
parser.add_argument('--probe-jitter-scale', type=float, default=0.15,
                    help='jitter scale relative to running data std')
parser.add_argument('--probe-lowfreq-scale', type=float, default=0.25,
                    help='lowfreq noise scale relative to running data std')
parser.add_argument('--probe-lowfreq-size', type=int, default=4,
                    help='base resolution for low-frequency noise (will be upsampled to image size)')
parser.add_argument('--probe-clamp', type=float, default=4.0,
                    help='clamp probe tensor values to [-probe-clamp, probe-clamp] to avoid extreme out-of-range probes')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Dataset import
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
# Helpers: debias schedule
# -------------------------
def debias_weight(epoch: int) -> float:
    """Linear ramp from 0 to 1 after debiasstart."""
    if epoch <= args.debiasstart:
        return 0.0
    if args.debias_ramp <= 0:
        return 1.0
    # epoch = debiasstart+1 -> 1/ramp ; ... -> 1
    w = (epoch - args.debiasstart) / float(args.debias_ramp)
    return float(min(1.0, max(0.0, w)))


# -------------------------
# Scheme-1: Probe state (data stats + bias EMA) and sampler
# -------------------------
class ProbeState:
    """
    Maintains:
      - running data mean/std (in *model input tensor space*, i.e., after transforms)
      - bias EMA for different model keys (e.g., 'train' vs 'ema')
    Provides:
      - sample_probes(K, H, W)
      - estimate_bias(model, H, W, key) with stable aggregation:
            b = center(log(mean_k softmax(z(I_k))))
    """
    def __init__(self, num_class: int, device: torch.device):
        self.num_class = num_class
        self.device = device

        self.data_mean = None  # shape [1,C,1,1]
        self.data_std = None   # shape [1,C,1,1]
        self.data_initialized = False

        self.bias_ema = {}  # key -> tensor [num_class]
        self.last_metrics = {}  # key -> dict

    @torch.no_grad()
    def update_data_stats(self, x: torch.Tensor):
        """Update running mean/std using EMA. x: [B,C,H,W] in model input space."""
        if x is None:
            return
        mu = x.mean(dim=(0, 2, 3), keepdim=True)  # [1,C,1,1]
        sd = x.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)

        if not self.data_initialized:
            self.data_mean = mu.detach().clone()
            self.data_std = sd.detach().clone()
            self.data_initialized = True
            return

        m = float(args.data_stat_ema)
        self.data_mean.mul_(m).add_(mu * (1.0 - m))
        self.data_std.mul_(m).add_(sd * (1.0 - m))

    @torch.no_grad()
    def _lowfreq_noise(self, K: int, C: int, H: int, W: int) -> torch.Tensor:
        """
        Generate low-frequency noise by sampling at low resolution and upsampling.
        Returns: [K,C,H,W], per-sample standardized (zero-mean, unit-std).
        """
        base = max(2, int(args.probe_lowfreq_size))
        noise = torch.randn(K, C, base, base, device=self.device)
        noise = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)

        # standardize each sample & channel
        mean = noise.mean(dim=(2, 3), keepdim=True)
        std = noise.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        return (noise - mean) / std

    @torch.no_grad()
    def sample_probes(self, K: int, H: int, W: int) -> torch.Tensor:
        """
        Mixture distribution Q:
          - const(mean)
          - jitter(mean + gaussian)
          - lowfreq(mean + upsampled lowfreq noise)
        All in model input space.
        """
        if not self.data_initialized:
            # fall back to zeros (usually corresponds to dataset mean after normalization)
            C = 3
            return torch.zeros(K, C, H, W, device=self.device)

        C = int(self.data_mean.shape[1])
        mu = self.data_mean  # [1,C,1,1]
        sd = self.data_std   # [1,C,1,1]

        # mixture decisions
        u = torch.rand(K, device=self.device)
        p_const = float(args.probe_mix_const)
        p_jitter = float(args.probe_mix_jitter)
        # remaining -> lowfreq

        probes = torch.empty(K, C, H, W, device=self.device)

        # 1) const
        mask_const = (u < p_const)
        if mask_const.any():
            probes[mask_const] = mu.expand(mask_const.sum(), C, H, W)

        # 2) jitter
        mask_jitter = (u >= p_const) & (u < p_const + p_jitter)
        if mask_jitter.any():
            eps = torch.randn(mask_jitter.sum(), C, H, W, device=self.device)
            probes[mask_jitter] = mu.expand(mask_jitter.sum(), C, H, W) + eps * (sd * float(args.probe_jitter_scale))

        # 3) lowfreq
        mask_low = ~(mask_const | mask_jitter)
        if mask_low.any():
            lf = self._lowfreq_noise(mask_low.sum(), C, H, W)
            probes[mask_low] = mu.expand(mask_low.sum(), C, H, W) + lf * (sd * float(args.probe_lowfreq_scale))

        # clamp to avoid extreme activations
        clamp_v = float(args.probe_clamp)
        if clamp_v > 0:
            probes = probes.clamp(min=-clamp_v, max=clamp_v)
        return probes

    @torch.no_grad()
    def estimate_bias(self, model: nn.Module, H: int, W: int, key: str = 'train'):
        """
        Estimate bias vector b in R^{num_class}:
            p_bar = mean_k softmax(z(I_k))
            b_raw = center(log(p_bar))
        Then apply EMA over time: b = EMA(b_raw).
        Also returns metrics (entropy, var, norm).
        """
        if key not in self.bias_ema:
            self.bias_ema[key] = torch.zeros(self.num_class, device=self.device)

        K = int(args.probe_k)
        probes = self.sample_probes(K, H, W)

        # prevent BN pollution by default
        prev_mode = model.training
        if args.probe_mode == 'eval':
            model.eval()
        else:
            model.train()

        logits, _ = model(probes)  # [K, num_class]
        # back to original mode
        model.train(prev_mode)

        probs = F.softmax(logits.float(), dim=1)  # [K,C]
        p_bar = probs.mean(dim=0).clamp(min=1e-8)  # [C]
        log_p = torch.log(p_bar)
        b_raw = log_p - log_p.mean()  # centering -> sum(b)=0

        # EMA smoothing for b
        ema = float(args.probe_ema)
        self.bias_ema[key].mul_(ema).add_(b_raw * (1.0 - ema))

        # metrics
        entropy = float((-p_bar * torch.log(p_bar)).sum().item())
        # per-probe centered log-prob variance (rough stability proxy)
        per = torch.log(probs.clamp(min=1e-8))
        per = per - per.mean(dim=1, keepdim=True)
        var = float(per.var(dim=0, unbiased=False).mean().item())
        norm = float(self.bias_ema[key].norm(p=2).item())

        self.last_metrics[key] = {'entropy': entropy, 'var': var, 'norm': norm}
        return self.bias_ema[key].detach(), self.last_metrics[key]


# -------------------------
# Main
# -------------------------
def main():
    global best_acc

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

    labeled_trainloader = data.DataLoader(
        train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    unlabeled_trainloader = data.DataLoader(
        train_unlabeled_set, batch_size=int(args.unlabeledratio * args.batch_size),
        shuffle=True, num_workers=4, drop_last=True
    )
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

    # Probe state (shared across epochs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probe_state = ProbeState(num_class=num_class, device=device)

    # Resume
    title = 'fixcdmad-probe1-' + args.dataset
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # optional: restore probe_state stats if present
        if 'probe_data_mean' in checkpoint and checkpoint['probe_data_mean'] is not None:
            probe_state.data_mean = checkpoint['probe_data_mean'].to(device)
            probe_state.data_std = checkpoint['probe_data_std'].to(device)
            probe_state.data_initialized = True
        if 'probe_bias_ema_train' in checkpoint:
            probe_state.bias_ema['train'] = checkpoint['probe_bias_ema_train'].to(device)
        if 'probe_bias_ema_ema' in checkpoint:
            probe_state.bias_ema['ema'] = checkpoint['probe_bias_ema_ema'].to(device)
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        # Keep the original 5-column logger interface to avoid breaking your parsing scripts.
        # Columns: bACC (no debias), GM (no debias), bACC (debias), GM (debias), Top1 (no debias)
        logger.set_names(['bACC', 'GM', 'bACC_debias', 'GM_debias', 'Top1'])

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f | DebiasW: %.3f' %
              (epoch + 1, args.epochs, state['lr'], debias_weight(epoch)))

        train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
              ema_optimizer, train_criterion, epoch, probe_state)

        test_acc1, testclassacc1, test_acc2, testclassacc2 = validate(
            test_loader, ema_model, criterion, mode='Test Stats ', epoch=epoch, probe_state=probe_state
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
            # probe states
            'probe_data_mean': probe_state.data_mean,
            'probe_data_std': probe_state.data_std,
            'probe_bias_ema_train': probe_state.bias_ema.get('train', None),
            'probe_bias_ema_ema': probe_state.bias_ema.get('ema', None),
        }, epoch + 1)

    logger.close()


def geometric_mean(accperclass: np.ndarray) -> float:
    """Geometric mean of per-class accuracies, with tiny floor to avoid zeros."""
    gm = 1.0
    for i in range(num_class):
        if accperclass[i] == 0:
            gm *= (1 / (100 * num_class)) ** (1 / num_class)
        else:
            gm *= (accperclass[i]) ** (1 / num_class)
    return float(gm)


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, probe_state: ProbeState):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()

    # probe metrics (epoch average)
    probe_entropy_m = AverageMeter()
    probe_var_m = AverageMeter()
    probe_norm_m = AverageMeter()

    end = time.time()
    bar = Bar('Training', max=args.val_iteration)

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    dW = debias_weight(epoch)

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

        # update running data stats for probe sampling (use labeled + weak unlabeled)
        probe_state.update_data_stats(inputs_x)
        probe_state.update_data_stats(inputs_u)

        H, W = int(inputs_x.shape[2]), int(inputs_x.shape[3])

        with torch.no_grad():
            outputs_u, _ = model(inputs_u)  # weak unlabeled logits

            # Scheme-1 debias: estimate bias using a *distribution* of probes
            # Update bias every probe_update_freq iterations (only when debias is active)
            if dW > 0 and (batch_idx % int(args.probe_update_freq) == 0):
                b, m = probe_state.estimate_bias(model, H=H, W=W, key='train')
                probe_entropy_m.update(m['entropy'], 1)
                probe_var_m.update(m['var'], 1)
                probe_norm_m.update(m['norm'], 1)
            else:
                b = probe_state.bias_ema.get('train', torch.zeros(num_class, device=inputs_x.device))

            if dW > 0:
                outputs_u = outputs_u - dW * b.view(1, -1).detach()

            targets_u2 = F.softmax(outputs_u, dim=1).detach()

        max_p, p_hat = torch.max(targets_u2, dim=1)
        p_hat = torch.zeros(int(args.unlabeledratio * batch_size), num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
        select_mask = max_p.ge(args.tau)
        select_mask = torch.cat([select_mask, select_mask], 0).float()

        # Use soft targets (your original code uses targets_u2 rather than p_hat)
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

        # plot progress
        suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | ETA: {eta:} | Loss: {loss:.4f} | Lx: {loss_x:.4f} | Lu: {loss_u:.4f}'.format(
            batch=batch_idx + 1, size=args.val_iteration, data=data_time.avg, bt=batch_time.avg, eta=bar.eta_td,
            loss=losses.avg, loss_x=losses_x.avg, loss_u=losses_u.avg
        )
        if dW > 0 and probe_entropy_m.count > 0:
            suffix += ' | ProbeH: %.3f Var: %.4f ||b||: %.3f' % (probe_entropy_m.avg, probe_var_m.avg, probe_norm_m.avg)
        bar.suffix = suffix
        bar.next()

    bar.finish()
    return (losses.avg, losses_x.avg, losses_u.avg)


def validate(valloader, model, criterion, mode, epoch: int, probe_state: ProbeState):
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

    # Estimate test-time bias once (using probe distribution) on EMA model
    # Use the last known H,W from the first batch.
    with torch.no_grad():
        first_batch = next(iter(valloader))
        inputs0 = first_batch[0].cuda()
        H, W = int(inputs0.shape[2]), int(inputs0.shape[3])
        # Update probe data stats using this batch as well (slightly improves representativeness for test-time)
        probe_state.update_data_stats(inputs0)
        b_ema, _m = probe_state.estimate_bias(model, H=H, W=W, key='ema')

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            outputs, _ = model(inputs)
            outputs2 = outputs - b_ema.view(1, -1)

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
            # customized weight decay
            param.mul_(1 - self.wd)


if __name__ == '__main__':
    main()
