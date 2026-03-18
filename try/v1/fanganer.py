# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP)
# Modified (Scheme-2): bias decomposition (intrinsic bias b0 vs training-induced drift Δb_t)
#
# Key idea:
#   - Use a single baseline picture (white image) to probe logit bias.
#   - Measure intrinsic bias once at epoch=0: b0 = z_theta(I_white) (centered).
#   - During training, measure current bias bt = z_theta(I_white) (centered),
#       and only subtract the drift part: Δb_t = bt - b0.
#   - Debias strength starts after --debiasstart with optional linear ramp.
#
# NOTE: This script is intentionally independent from Scheme-1 (no probe distributions / no data-stat probes).

from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np

# restore np.int for legacy code
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
parser = argparse.ArgumentParser(description='PyTorch FixMatch Training (CDMAD baseline, Scheme-2 bias decomposition)')

# Optimization
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.0015, type=float)

# Checkpoints / output
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--out', default='result', type=str)

# Seeds / device
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

# Dataset
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10/cifar100/stl10')
parser.add_argument('--imbalancetype', type=str, default='long', help='long or step imbalance')
parser.add_argument('--unlabeledratio', type=float, default=2)

# Scheme-2: Debias schedule
parser.add_argument('--debiasstart', type=int, default=100, help='epoch to start drift debias')
parser.add_argument('--debias-ramp', type=int, default=20, help='linear ramp epochs after debiasstart (0 -> immediate full)')
parser.add_argument('--delta-ema', type=float, default=0.99, help='EMA for Δb_t (drift bias)')
parser.add_argument('--bias-mode', type=str, default='eval', choices=['eval', 'train'],
                    help='forward baseline picture under model.eval() (recommended) or model.train()')
parser.add_argument('--baseline-value', type=float, default=1.0, help='pixel value for baseline picture (white=1.0)')
parser.add_argument('--baseline-batch', type=int, default=1, help='baseline picture batch size (kept 1 by default)')

# Scheme-2: What to subtract at test-time
parser.add_argument('--test-debias', type=str, default='full', choices=['none', 'delta', 'full'],
                    help='test-time debias: none / delta(Δb_t) / full(bt)')

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
# Schedule helpers
# -------------------------
def ramp_weight(epoch: int) -> float:
    """Linear ramp for debias strength after debiasstart."""
    if epoch <= args.debiasstart:
        return 0.0
    if args.debias_ramp <= 0:
        return 1.0
    w = (epoch - args.debiasstart) / float(args.debias_ramp)
    return float(min(1.0, max(0.0, w)))


# -------------------------
# Scheme-2: Bias decomposer
# -------------------------
class BiasDecomposer:
    """
    Scheme-2 decomposes bias probed by a baseline picture into:
      - b0: intrinsic bias at epoch=0
      - Δb_t: training-induced drift bias
    We estimate bt by forwarding a single baseline picture I0 (white by default),
    centering logits to remove additive constant, then:
        bt = center(z(I0))
        Δb_t = bt - b0
    We maintain EMA for Δb_t for stability.

    NOTE: This is intentionally simple and independent: no probe distributions.
    """
    def __init__(self, num_class: int, device: torch.device):
        self.num_class = num_class
        self.device = device
        self.b0 = torch.zeros(num_class, device=device)
        self.delta_ema = torch.zeros(num_class, device=device)
        self.initialized = False

    @staticmethod
    def _center(v: torch.Tensor) -> torch.Tensor:
        return v - v.mean()

    @torch.no_grad()
    def _forward_baseline(self, model: nn.Module, H: int, W: int) -> torch.Tensor:
        """Forward baseline picture and return centered logits: center(z(I0))."""
        B = int(args.baseline_batch)
        val = float(args.baseline_value)
        baseline = torch.full((B, 3, H, W), fill_value=val, device=self.device)

        prev_train = model.training
        if args.bias_mode == 'eval':
            model.eval()
        else:
            model.train()

        logits, _ = model(baseline)  # [B, C]
        # restore
        model.train(prev_train)

        logits = logits.mean(dim=0).float()  # [C]
        return self._center(logits)

    @torch.no_grad()
    def init_b0(self, model: nn.Module, H: int, W: int):
        """Measure intrinsic bias b0 once (epoch=0, before training)."""
        self.b0 = self._forward_baseline(model, H, W).detach()
        self.delta_ema.zero_()
        self.initialized = True

    @torch.no_grad()
    def update(self, model: nn.Module, H: int, W: int):
        """Update Δb_t EMA based on current bt - b0."""
        if not self.initialized:
            self.init_b0(model, H, W)
            return self.delta_ema

        bt = self._forward_baseline(model, H, W)
        delta = (bt - self.b0)
        m = float(args.delta_ema)
        self.delta_ema.mul_(m).add_(delta * (1.0 - m))
        return self.delta_ema

    @torch.no_grad()
    def current_full_bias(self, model: nn.Module, H: int, W: int) -> torch.Tensor:
        """Return bt (centered) for current model."""
        if not self.initialized:
            self.init_b0(model, H, W)
        return self._forward_baseline(model, H, W)

    @torch.no_grad()
    def current_delta_bias(self, model: nn.Module, H: int, W: int) -> torch.Tensor:
        """Return Δb_t EMA for current model."""
        return self.update(model, H, W)


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

    # Scheme-2 bias decomposer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bias_dec = BiasDecomposer(num_class=num_class, device=device)

    # Resume
    title = 'fixcdmad-scheme2-' + args.dataset
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # restore scheme-2 states if present
        if 'b0' in checkpoint:
            bias_dec.b0 = checkpoint['b0'].to(device)
            bias_dec.delta_ema = checkpoint.get('delta_ema', torch.zeros(num_class)).to(device)
            bias_dec.initialized = True
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        # Keep a stable 5-column layout compatible with your previous parsing
        logger.set_names(['bACC', 'GM', 'bACC_debias', 'GM_debias', 'Top1'])

    # Initialize b0 using a real batch's shape (handles STL sizes)
    with torch.no_grad():
        x0, _, _ = next(iter(labeled_trainloader))
        H0, W0 = int(x0.shape[2]), int(x0.shape[3])
        bias_dec.init_b0(model, H0, W0)
        print('[Scheme-2] Initialized b0 (intrinsic) | ||b0||=%.4f' % bias_dec.b0.norm(p=2).item())

    for epoch in range(start_epoch, args.epochs):
        w = ramp_weight(epoch)
        print('\nEpoch: [%d | %d] LR: %f | DriftDebiasW: %.3f' %
              (epoch + 1, args.epochs, state['lr'], w))

        train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer,
              train_criterion, epoch, bias_dec)

        test_acc1, testclassacc1, test_acc2, testclassacc2 = validate(
            test_loader, ema_model, criterion, mode='Test Stats ', epoch=epoch, bias_dec=bias_dec
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
            # scheme-2 states
            'b0': bias_dec.b0.detach().cpu(),
            'delta_ema': bias_dec.delta_ema.detach().cpu(),
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


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, bias_dec: BiasDecomposer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()

    # drift-bias metrics (optional)
    delta_norm_m = AverageMeter()

    end = time.time()
    bar = Bar('Training', max=args.val_iteration)

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    w = ramp_weight(epoch)

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

        with torch.no_grad():
            outputs_u, _ = model(inputs_u)  # weak logits

            if w > 0.0:
                # Scheme-2: subtract only drift bias Δb_t (EMA)
                delta = bias_dec.update(model, H=H, W=W)
                delta_norm_m.update(delta.norm(p=2).item(), 1)
                outputs_u = outputs_u - w * delta.view(1, -1).detach()

            targets_u2 = F.softmax(outputs_u, dim=1).detach()

        max_p, p_hat = torch.max(targets_u2, dim=1)
        p_hat = torch.zeros(int(args.unlabeledratio * batch_size), num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
        select_mask = max_p.ge(args.tau)
        select_mask = torch.cat([select_mask, select_mask], 0).float()

        # Keep your original choice: use soft targets_u2 rather than hard p_hat
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

        suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | ETA: {eta:} | Loss: {loss:.4f} | Lx: {loss_x:.4f} | Lu: {loss_u:.4f}'.format(
            batch=batch_idx + 1, size=args.val_iteration, data=data_time.avg, bt=batch_time.avg, eta=bar.eta_td,
            loss=losses.avg, loss_x=losses_x.avg, loss_u=losses_u.avg
        )
        if w > 0.0 and delta_norm_m.count > 0:
            suffix += ' | ||Δb||: %.3f' % (delta_norm_m.avg,)
        bar.suffix = suffix
        bar.next()

    bar.finish()
    return (losses.avg, losses_x.avg, losses_u.avg)


def validate(valloader, model, criterion, mode, epoch: int, bias_dec: BiasDecomposer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
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
        # infer input size from first batch
        first_batch = next(iter(valloader))
        x0 = first_batch[0].cuda()
        H, W = int(x0.shape[2]), int(x0.shape[3])

        # compute test-time bias vector according to args.test_debias
        if args.test_debias == 'none':
            b_test = torch.zeros(num_class, device=x0.device)
        elif args.test_debias == 'delta':
            # Use drift-only bias for this (EMA update on EMA model)
            b_test = bias_dec.current_delta_bias(model, H=H, W=W)
        else:
            # full bt bias
            b_test = bias_dec.current_full_bias(model, H=H, W=W)

        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            outputs, _ = model(inputs)
            outputs2 = outputs - b_test.view(1, -1)

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
                batch=batch_idx + 1, size=len(valloader), data=data_time.avg, bt=batch_time.avg, top1=top1.avg, top5=top5.avg
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
