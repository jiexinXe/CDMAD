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
# Scheme-1 (Minimal): Anchor-augmented probe distribution + bias scale
# -------------------------
# Only TWO additional knobs are exposed for Scheme-1:
#   --probe-anchor : fraction of probes that are CDMAD-like solid (constant) images (anchor)
#   --bias-scale   : global scaling for the debias vector (shrinkage / strength control)
parser.add_argument('--probe-anchor', type=float, default=0.15,
                    help='Fraction of probes to use as a solid constant anchor image (CDMAD-like). Range [0,1].')
parser.add_argument('--bias-scale', type=float, default=1.0,
                    help='Global scaling for the estimated bias vector before subtraction (strength control).')


args = parser.parse_args()

# -------------------------
# Scheme-1 fixed implementation constants (kept internal to avoid too many hyper-parameters)
# -------------------------
PROBE_K = 16
PROBE_UPDATE_FREQ = 20     # update bias every N iterations when debias is active (stabilized)
PROBE_MODE = 'eval'        # 'eval' recommended to avoid BN pollution
PROBE_EMA = 0.99           # EMA decay for bias vector
PROBE_WARMUP_EPOCHS = 20    # warm up probe bias estimation before debiasstart (no extra args)
DATA_STAT_EMA = 0.99       # EMA decay for running data mean/std
# Mixture for the non-anchor portion (kept fixed)
MIX_CONST = 0.40           # constant probe = running data mean
MIX_JITTER = 0.30          # jittered mean (mean + gaussian)
JITTER_SCALE = 0.15        # relative to running std
LOWFREQ_SCALE = 0.25       # relative to running std
LOWFREQ_SIZE = 4           # low-frequency base resolution
PROBE_CLAMP = 4.0          # clamp probe tensor values to [-PROBE_CLAMP, PROBE_CLAMP]
SOLID_VALUE = 1.0          # anchor image value in normalized input space (CDMAD-like "white" constant)

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
    """Step function: full debias after debiasstart (no ramp)."""
    return 1.0 if epoch > args.debiasstart else 0.0


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

        self.bias_std_ref = {}  # key -> scalar tensor (reference std for scale-decoupling)
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

        m = float(DATA_STAT_EMA)
        self.data_mean.mul_(m).add_(mu * (1.0 - m))
        self.data_std.mul_(m).add_(sd * (1.0 - m))

    @torch.no_grad()
    def _lowfreq_noise(self, K: int, C: int, H: int, W: int) -> torch.Tensor:
        """
        Generate low-frequency noise by sampling at low resolution and upsampling.
        Returns: [K,C,H,W], per-sample standardized (zero-mean, unit-std).
        """
        base = max(2, int(LOWFREQ_SIZE))
        noise = torch.randn(K, C, base, base, device=self.device)
        noise = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)

        # standardize each sample & channel
        mean = noise.mean(dim=(2, 3), keepdim=True)
        std = noise.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        return (noise - mean) / std

    @torch.no_grad()
    def sample_probes(self, K: int, H: int, W: int) -> torch.Tensor:
        """
        Probe distribution Q (standalone Scheme-1, minimal knobs):

        Q = anchor * Q_solid  + (1-anchor) * Q_mix

        - Q_solid: CDMAD-like solid constant image (value = SOLID_VALUE in normalized input space)
        - Q_mix  : fixed mixture in model input space:
            * const(mean) with prob MIX_CONST
            * jitter(mean + gaussian) with prob MIX_JITTER
            * lowfreq(mean + upsampled lowfreq noise) with remaining prob
        """
        if not self.data_initialized:
            # fall back to zeros (usually corresponds to dataset mean after normalization)
            C = 3
            return torch.zeros(K, C, H, W, device=self.device)

        C = int(self.data_mean.shape[1])
        mu = self.data_mean  # [1,C,1,1]
        sd = self.data_std   # [1,C,1,1]

        anchor = float(args.probe_anchor)
        anchor = max(0.0, min(1.0, anchor))
        K_anchor = int(round(K * anchor))
        K_anchor = max(0, min(K_anchor, K))
        K_rest = K - K_anchor

        probes = torch.empty(K, C, H, W, device=self.device)

        # 0) solid anchor (CDMAD-like)
        if K_anchor > 0:
            probes[:K_anchor] = torch.full((K_anchor, C, H, W), float(SOLID_VALUE), device=self.device)

        # 1) mixture (non-anchor)
        if K_rest > 0:
            u = torch.rand(K_rest, device=self.device)
            p_const = float(MIX_CONST)
            p_jitter = float(MIX_JITTER)

            sub = probes[K_anchor:]  # view
            # const(mean)
            mask_const = (u < p_const)
            if mask_const.any():
                sub[mask_const] = mu.expand(mask_const.sum(), C, H, W)

            # jitter(mean + gaussian)
            mask_jitter = (u >= p_const) & (u < p_const + p_jitter)
            if mask_jitter.any():
                eps = torch.randn(mask_jitter.sum(), C, H, W, device=self.device)
                sub[mask_jitter] = mu.expand(mask_jitter.sum(), C, H, W) + eps * (sd * float(JITTER_SCALE))

            # lowfreq(mean + lowfreq noise)
            mask_low = ~(mask_const | mask_jitter)
            if mask_low.any():
                lf = self._lowfreq_noise(mask_low.sum(), C, H, W)
                sub[mask_low] = mu.expand(mask_low.sum(), C, H, W) + lf * (sd * float(LOWFREQ_SCALE))

        # clamp to avoid extreme activations
        clamp_v = float(PROBE_CLAMP)
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

        K = int(PROBE_K)
        probes = self.sample_probes(K, H, W)

        # prevent BN pollution by default
        prev_mode = model.training
        if PROBE_MODE == 'eval':
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

        # --- scale-decoupled bias: keep std(b) stable over training ---
        cur_std = b_raw.std(unbiased=False).clamp(min=1e-6)
        if key not in self.bias_std_ref:
            # set reference scale on first update for this key
            self.bias_std_ref[key] = cur_std.detach().clone()
        ref_std = self.bias_std_ref[key]
        b_scaled = b_raw / cur_std * ref_std

        # EMA smoothing for b (vector)
        ema = float(PROBE_EMA)
        self.bias_ema[key].mul_(ema).add_(b_scaled * (1.0 - ema))
        # metrics
        entropy = float((-p_bar * torch.log(p_bar)).sum().item())
        # per-probe centered log-prob variance (rough stability proxy)
        per = torch.log(probs.clamp(min=1e-8))
        per = per - per.mean(dim=1, keepdim=True)
        var = float(per.var(dim=0, unbiased=False).mean().item())
        norm = float(self.bias_ema[key].norm(p=2).item())

        self.last_metrics[key] = {
            'entropy': entropy,
            'var': var,
            'norm': norm,
            'std_raw': float(cur_std.item()),
            'std_ref': float(ref_std.item()),
            'std_ema': float(self.bias_ema[key].std(unbiased=False).item()),
        }
        return self.bias_ema[key].detach(), self.last_metrics[key]


# -------------------------
# Main
# -------------------------
def main():
    global best_acc

    # Best-checkpoint trackers (debiased metrics)
    best_top1_debias = 0.0
    best_top1_debias_epoch = 0
    best_gm_debias = 0.0
    best_gm_debias_epoch = 0

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio, args.imbalancetype)
    U_SAMPLES_PER_CLASS = make_imb_data(args.num_max_u, num_class, args.imb_ratio_u, args.imbalancetype)

    # Head/Mid/Tail class splits (by labeled sample counts)
    head_idx, mid_idx, tail_idx = split_head_mid_tail(N_SAMPLES_PER_CLASS)

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
        if 'probe_bias_std_ref_train' in checkpoint and checkpoint['probe_bias_std_ref_train'] is not None:
            probe_state.bias_std_ref['train'] = checkpoint['probe_bias_std_ref_train'].to(device)
        if 'probe_bias_std_ref_ema' in checkpoint and checkpoint['probe_bias_std_ref_ema'] is not None:
            probe_state.bias_std_ref['ema'] = checkpoint['probe_bias_std_ref_ema'].to(device)
        # restore best metrics if present
        if 'best_top1_debias' in checkpoint:
            best_top1_debias = float(checkpoint['best_top1_debias'])
            best_top1_debias_epoch = int(checkpoint.get('best_top1_debias_epoch', 0))
        if 'best_gm_debias' in checkpoint:
            best_gm_debias = float(checkpoint['best_gm_debias'])
            best_gm_debias_epoch = int(checkpoint.get('best_gm_debias_epoch', 0))
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        # Keep the original 5-column logger interface to avoid breaking your parsing scripts.
        # Columns: bACC (no debias), GM (no debias), bACC (debias), GM (debias), Top1 (no debias)
        logger.set_names(['bACC', 'GM', 'bACC_debias', 'GM_debias', 'Top1', 'Top1_debias'])

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f | DebiasW: %.3f' %
              (epoch + 1, args.epochs, state['lr'], debias_weight(epoch)))

        train_stats = train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
              ema_optimizer, train_criterion, epoch, probe_state)

        # Diagnostics CSV (appends one row per epoch)
        diag_path = os.path.join(args.out, 'diag_scheme1.csv')
        if (epoch == start_epoch) and (not os.path.exists(diag_path)):
            with open(diag_path, 'w') as f:
                f.write('epoch,dW,loss,loss_x,loss_u,mask_rate,max_p,probe_entropy,probe_var,bias_norm,bias_std_raw,bias_std_ref,bias_std_ema,corr_norm,corr_maxabs\n')
        with open(diag_path, 'a') as f:
            f.write(f"{epoch+1},{train_stats['dW']:.6f},{train_stats['loss']:.6f},{train_stats['loss_x']:.6f},{train_stats['loss_u']:.6f},"
                    f"{train_stats['mask_rate']:.6f},{train_stats['max_p']:.6f},{train_stats['probe_entropy']:.6f},{train_stats['probe_var']:.6f},{train_stats['bias_norm']:.6f},"
                    f"{train_stats['bias_std_raw']:.6f},{train_stats['bias_std_ref']:.6f},{train_stats['bias_std_ema']:.6f},"
                    f"{train_stats['corr_norm']:.6f},{train_stats['corr_maxabs']:.6f}\n")

        test_acc1, testclassacc1, test_acc2, testclassacc2 = validate(
            test_loader, ema_model, criterion, mode='Test Stats ', epoch=epoch, probe_state=probe_state
        )

        GM = geometric_mean(testclassacc1)
        GM2 = geometric_mean(testclassacc2)

        # Head/Mid/Tail (by labeled counts) for raw & debiased
        head_raw = float(np.mean(testclassacc1[head_idx]))
        mid_raw = float(np.mean(testclassacc1[mid_idx]))
        tail_raw = float(np.mean(testclassacc1[tail_idx]))
        head_deb = float(np.mean(testclassacc2[head_idx]))
        mid_deb = float(np.mean(testclassacc2[mid_idx]))
        tail_deb = float(np.mean(testclassacc2[tail_idx]))

        # Best checkpointing (based on debiased metrics)
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'probe_data_mean': probe_state.data_mean,
            'probe_data_std': probe_state.data_std,
            'probe_bias_ema_train': probe_state.bias_ema.get('train', None),
            'probe_bias_ema_ema': probe_state.bias_ema.get('ema', None),
            'probe_bias_std_ref_train': probe_state.bias_std_ref.get('train', None),
            'probe_bias_std_ref_ema': probe_state.bias_std_ref.get('ema', None),
            'best_top1_debias': best_top1_debias,
            'best_top1_debias_epoch': best_top1_debias_epoch,
            'best_gm_debias': best_gm_debias,
            'best_gm_debias_epoch': best_gm_debias_epoch,
        }
        if test_acc2 > best_top1_debias:
            best_top1_debias = float(test_acc2)
            best_top1_debias_epoch = int(epoch + 1)
            state['best_top1_debias'] = best_top1_debias
            state['best_top1_debias_epoch'] = best_top1_debias_epoch
            save_best_checkpoint(state, args.out, 'best_top1_debias.pth.tar')
        if GM2 > best_gm_debias:
            best_gm_debias = float(GM2)
            best_gm_debias_epoch = int(epoch + 1)
            state['best_gm_debias'] = best_gm_debias
            state['best_gm_debias_epoch'] = best_gm_debias_epoch
            save_best_checkpoint(state, args.out, 'best_gm_debias.pth.tar')
        # Full diagnostics CSV (train + test, one row per epoch)
        full_path = os.path.join(args.out, 'diag_scheme1_full.csv')
        if (epoch == start_epoch) and (not os.path.exists(full_path)):
            with open(full_path, 'w') as f:
                f.write('epoch,dW,loss,loss_x,loss_u,mask_rate,max_p,probe_entropy,probe_var,bias_norm,bias_std_raw,bias_std_ref,bias_std_ema,corr_norm,corr_maxabs,'
                        'Top1,Top1_debias,bACC,GM,bACC_debias,GM_debias,head_raw,mid_raw,tail_raw,head_debias,mid_debias,tail_debias,'
                        'best_top1_debias,best_top1_debias_epoch,best_gm_debias,best_gm_debias_epoch\n')
        with open(full_path, 'a') as f:
            f.write(
                f"{epoch+1},{train_stats['dW']:.6f},{train_stats['loss']:.6f},{train_stats['loss_x']:.6f},{train_stats['loss_u']:.6f},"
                f"{train_stats['mask_rate']:.6f},{train_stats['max_p']:.6f},{train_stats['probe_entropy']:.6f},{train_stats['probe_var']:.6f},{train_stats['bias_norm']:.6f},"
                f"{train_stats['bias_std_raw']:.6f},{train_stats['bias_std_ref']:.6f},{train_stats['bias_std_ema']:.6f},{train_stats['corr_norm']:.6f},{train_stats['corr_maxabs']:.6f},"
                f"{test_acc1:.6f},{test_acc2:.6f},{float(testclassacc1.mean()):.6f},{GM:.6f},{float(testclassacc2.mean()):.6f},{GM2:.6f},"
                f"{head_raw:.6f},{mid_raw:.6f},{tail_raw:.6f},{head_deb:.6f},{mid_deb:.6f},{tail_deb:.6f},"
                f"{best_top1_debias:.6f},{best_top1_debias_epoch},{best_gm_debias:.6f},{best_gm_debias_epoch}\n"
            )

        print("raw Test Top1:", test_acc1, " | debiased Test Top1:", test_acc2)
        print("without test debias bACC:", testclassacc1.mean(), "GM:", GM,
              "with test debias bACC:", testclassacc2.mean(), "GM:", GM2)

        logger.append([testclassacc1.mean(), GM, testclassacc2.mean(), GM2, test_acc1, test_acc2])

        save_checkpoint(state, epoch + 1)

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
    probe_std_raw_m = AverageMeter()
    probe_std_ref_m = AverageMeter()
    probe_std_ema_m = AverageMeter()
    corr_norm_m = AverageMeter()
    corr_maxabs_m = AverageMeter()

    # pseudo-label diagnostics (epoch average)
    mask_rate_m = AverageMeter()
    max_p_m = AverageMeter()

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
            # Update bias every PROBE_UPDATE_FREQ iterations.
            # Warm-start bias estimation PROBE_WARMUP_EPOCHS epochs before debiasstart, so turning debias on doesn't shock training.
            bias_warm_epoch = max(0, int(args.debiasstart) - int(PROBE_WARMUP_EPOCHS))
            do_probe_update = (epoch >= bias_warm_epoch) and (batch_idx % int(PROBE_UPDATE_FREQ) == 0)
            if do_probe_update:
                # use EMA teacher for more stable probe statistics
                b, m = probe_state.estimate_bias(ema_optimizer.ema_model, H=H, W=W, key='train')
                probe_entropy_m.update(m['entropy'], 1)
                probe_var_m.update(m['var'], 1)
                probe_norm_m.update(m['norm'], 1)
                probe_std_raw_m.update(m.get('std_raw', 0.0), 1)
                probe_std_ref_m.update(m.get('std_ref', 0.0), 1)
                probe_std_ema_m.update(m.get('std_ema', 0.0), 1)
            else:
                b = probe_state.bias_ema.get('train', torch.zeros(num_class, device=inputs_x.device))

            b_eff = b * float(args.bias_scale)
            if dW > 0:
                corr_vec = (dW * b_eff).detach()
                corr_norm_m.update(float(corr_vec.norm(p=2).item()), 1)
                corr_maxabs_m.update(float(corr_vec.abs().max().item()), 1)
            outputs_u = outputs_u - dW * b_eff.view(1, -1).detach()

            targets_u2 = F.softmax(outputs_u, dim=1).detach()

        max_p, p_hat = torch.max(targets_u2, dim=1)
        p_hat = torch.zeros(int(args.unlabeledratio * batch_size), num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
        select_mask = max_p.ge(args.tau)
        select_mask = torch.cat([select_mask, select_mask], 0).float()

        # diagnostics for pseudo-labeling
        mask_rate_m.update(float(max_p.ge(args.tau).float().mean().item()), 1)
        max_p_m.update(float(max_p.mean().item()), 1)

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
            suffix += ' | ProbeH: %.3f Var: %.4f ||b||: %.3f | Corr||: %.3f CorrMax: %.3f' % (probe_entropy_m.avg, probe_var_m.avg, probe_norm_m.avg, corr_norm_m.avg, corr_maxabs_m.avg)
        bar.suffix = suffix
        bar.next()

    bar.finish()
    return {
        'loss': float(losses.avg),
        'loss_x': float(losses_x.avg),
        'loss_u': float(losses_u.avg),
        'dW': float(dW),
        'mask_rate': float(mask_rate_m.avg) if mask_rate_m.count > 0 else 0.0,
        'max_p': float(max_p_m.avg) if max_p_m.count > 0 else 0.0,
        'probe_entropy': float(probe_entropy_m.avg) if probe_entropy_m.count > 0 else 0.0,
        'probe_var': float(probe_var_m.avg) if probe_var_m.count > 0 else 0.0,
        'bias_norm': float(probe_norm_m.avg) if probe_norm_m.count > 0 else 0.0,
        'bias_std_raw': float(probe_std_raw_m.avg) if probe_std_raw_m.count > 0 else 0.0,
        'bias_std_ref': float(probe_std_ref_m.avg) if probe_std_ref_m.count > 0 else 0.0,
        'bias_std_ema': float(probe_std_ema_m.avg) if probe_std_ema_m.count > 0 else 0.0,
        'corr_norm': float(corr_norm_m.avg) if corr_norm_m.count > 0 else 0.0,
        'corr_maxabs': float(corr_maxabs_m.avg) if corr_maxabs_m.count > 0 else 0.0,
    }


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
            b_eff = b_ema * float(args.bias_scale)
            outputs2 = outputs - b_eff.view(1, -1)

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




def split_head_mid_tail(n_samples_per_class):
    """Return (head_idx, mid_idx, tail_idx) by labeled sample counts (descending)."""
    arr = np.asarray(n_samples_per_class, dtype=np.int64)
    order = np.argsort(-arr)  # desc
    c = len(arr)
    # 3-way split (rough thirds). For CIFAR10: 3/4/3; for CIFAR100: 33/34/33.
    head_n = c // 3
    tail_n = c // 3
    mid_n = c - head_n - tail_n
    head_idx = order[:head_n]
    mid_idx = order[head_n:head_n + mid_n]
    tail_idx = order[head_n + mid_n:]
    return head_idx, mid_idx, tail_idx

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


def save_best_checkpoint(state, checkpoint, filename):
    """Save a best checkpoint (no periodic copies)."""
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)



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