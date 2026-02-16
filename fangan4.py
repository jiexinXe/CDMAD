# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP)
# Modified (Scheme-4): Activation-matched & semantic-suppressed baseline pictures
#
# Key idea (Scheme-4, independent):
#   - Replace CDMAD's single white/solid reference with a small set of *learnable* probe images.
#   - Probes are optimized (few steps, periodically) to:
#       (A) match shallow activation statistics (BN pre-activation mean/var) of the current model
#       (B) maximize output entropy (semantic-suppressed)
#   - Bias logits are estimated as the average logits over these optimized probes, then used for debias:
#         outputs_u <- outputs_u - biaseddegree
#
# Independence note:
#   This script is intentionally independent from Scheme-1/2/3:
#     - no probe distribution mixtures (Scheme-1), no b0/Δb decomposition (Scheme-2),
#       no spectral FFT probes (Scheme-3).

from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np

# restore legacy np.int
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
parser = argparse.ArgumentParser(description='PyTorch FixMatch Training (CDMAD baseline, Scheme-4 activation-matched probes)')

# Optimization options
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
# Scheme-4: activation-matched probe options
# -------------------------
parser.add_argument('--am-num-probes', type=int, default=8, help='number of learnable probe images')
parser.add_argument('--am-bn-layers', type=int, default=3, help='use first N BatchNorm2d layers to match stats')
parser.add_argument('--am-update-freq', type=int, default=50, help='update probes every N iterations (global)')
parser.add_argument('--am-steps', type=int, default=5, help='gradient steps per probe update')
parser.add_argument('--am-lr', type=float, default=0.08, help='probe optimizer learning rate')
parser.add_argument('--am-clamp', type=float, default=4.0, help='clamp probe tensor values to [-am-clamp, am-clamp]')
parser.add_argument('--am-stat-weight', type=float, default=1.0, help='weight of activation-stat matching loss')
parser.add_argument('--am-ent-weight', type=float, default=0.2, help='weight of entropy (semantic-suppressed) loss')
parser.add_argument('--am-mode', type=str, default='eval', choices=['eval', 'train'],
                    help='forward probes under model.eval() (recommended) or model.train()')
parser.add_argument('--am-warmup-iters', type=int, default=200, help='warmup iters before using probes for debias')

# Validation adaptation (kept small; can be 0 to skip)
parser.add_argument('--am-test-adapt-steps', type=int, default=5, help='probe update steps on EMA model before test bias estimation')

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
# Scheme-4: Activation-matched probes
# -------------------------
class ActivationMatchedProbes:
    """
    Maintain learnable probes I (in model input tensor space). Optimize probes to:
      - match BN pre-activation running mean/var for early BN layers
      - maximize output entropy

    We capture BN *inputs* with forward_pre_hooks, so mean/var correspond to BN running_mean/var.
    """
    def __init__(self, model: nn.Module, device: torch.device, num_class: int):
        self.device = device
        self.num_class = num_class

        self.bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        if len(self.bn_layers) == 0:
            raise RuntimeError('No BatchNorm2d layers found; Scheme-4 requires BN in the backbone.')

        self.bn_layers = self.bn_layers[:max(1, int(args.am_bn_layers))]
        self._captured = {}
        self._capture_on = False
        self._hooks = []
        for idx, bn in enumerate(self.bn_layers):
            self._hooks.append(bn.register_forward_pre_hook(self._make_hook(idx)))

        self.probes = None   # nn.Parameter [M,3,H,W]
        self.opt = None      # Adam optimizer for probes
        self.HW = None
        self.global_iter = 0

    def _make_hook(self, idx):
        def _hook(module, inputs):
            if not self._capture_on:
                return
            # inputs[0]: [B,C,H,W]
            self._captured[idx] = inputs[0]
        return _hook

    def _ensure_shape(self, H: int, W: int):
        if self.probes is not None and self.HW == (H, W):
            return
        M = max(1, int(args.am_num_probes))
        # init near 0 (often corresponds to dataset mean in normalized space)
        init = torch.zeros((M, 3, H, W), device=self.device)
        # small noise to break symmetry
        init += 0.05 * torch.randn_like(init)
        self.probes = nn.Parameter(init)
        self.opt = optim.Adam([self.probes], lr=float(args.am_lr))
        self.HW = (H, W)

    @staticmethod
    def _entropy_loss_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """
        Minimize sum p log p -> maximize entropy.
        logits: [B,C]
        returns scalar
        """
        p = F.softmax(logits, dim=1).clamp(min=1e-8)
        return torch.mean(torch.sum(p * torch.log(p), dim=1))

    def _stat_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Compute activation-stat matching loss using captured BN inputs
        against BN running_mean/var.
        """
        loss = 0.0
        for idx, bn in enumerate(self.bn_layers):
            x = self._captured.get(idx, None)
            if x is None:
                continue
            # x: [B,C,H,W] pre-BN activation
            mu = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)

            target_mu = bn.running_mean.detach()
            target_var = bn.running_var.detach()

            # Align shapes
            if mu.shape != target_mu.shape:
                # should not happen, but guard anyway
                target_mu = target_mu.view_as(mu)
                target_var = target_var.view_as(var)

            loss = loss + F.mse_loss(mu, target_mu) + F.mse_loss(var, target_var)
        return loss

    def maybe_update(self, model: nn.Module, H: int, W: int, steps: int = None):
        """
        Update probes every am_update_freq iterations.
        Called during training; uses model in eval (default) to avoid BN pollution.
        """
        self._ensure_shape(H, W)

        self.global_iter += 1
        if steps is None:
            steps = int(args.am_steps)

        # Only update at scheduled frequency
        if int(args.am_update_freq) > 1 and (self.global_iter % int(args.am_update_freq) != 0):
            return

        # Run a few optimization steps
        prev_train = model.training
        if args.am_mode == 'eval':
            model.eval()
        else:
            model.train()

        for _ in range(max(1, steps)):
            self.opt.zero_grad(set_to_none=True)
            self._captured.clear()
            self._capture_on = True

            # Forward probes (need grad)
            logits, _ = model(self.probes)

            # capture done
            self._capture_on = False

            stat_loss = self._stat_loss(model) * float(args.am_stat_weight)
            ent_loss = self._entropy_loss_from_logits(logits) * float(args.am_ent_weight)
            loss = stat_loss + ent_loss

            loss.backward()
            self.opt.step()

            # clamp probes to reasonable range
            clamp_v = float(args.am_clamp)
            if clamp_v > 0:
                with torch.no_grad():
                    self.probes.clamp_(min=-clamp_v, max=clamp_v)

        # restore model mode
        model.train(prev_train)

    @torch.no_grad()
    def bias_logits(self, model: nn.Module, H: int, W: int) -> torch.Tensor:
        """
        Compute bias logits as mean logits over probes.
        """
        self._ensure_shape(H, W)
        prev_train = model.training
        model.eval()  # recommended for stability
        logits, _ = model(self.probes.detach())
        model.train(prev_train)
        return logits.mean(dim=0).detach()

    def close(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []


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
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_cifar10('./data', N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS, rand_number=args.manualSeed)
    elif args.dataset == 'stl10':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_stl10('./data', N_SAMPLES_PER_CLASS, args.out, rand_number=args.manualSeed)
    elif args.dataset == 'cifar100':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_cifar100('./data', N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS, rand_number=args.manualSeed)
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
    am_probes = ActivationMatchedProbes(model, device=device, num_class=num_class)
    am_probes_ema = ActivationMatchedProbes(ema_model, device=device, num_class=num_class)

    # Resume
    title = 'fixcdmad-scheme4-' + args.dataset
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        # stable 5 columns for your parsing scripts
        logger.set_names(['bACC', 'GM', 'bACC_debias', 'GM_debias', 'Top1'])

    # Determine H,W once (supports STL sizes)
    with torch.no_grad():
        x0, _, _ = next(iter(labeled_trainloader))
        H0, W0 = int(x0.shape[2]), int(x0.shape[3])
        # warmup update probes a bit before training loop to avoid "cold probes"
        for _ in range(3):
            am_probes.maybe_update(model, H0, W0, steps=max(1, int(args.am_steps)))

    global_iter = 0
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        global_iter = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer,
                            train_criterion, epoch, am_probes, global_iter)

        test_acc1, testclassacc1, test_acc2, testclassacc2 = validate(test_loader, ema_model, criterion,
                                                                      mode='Test Stats ', epoch=epoch,
                                                                      am_probes_ema=am_probes_ema)

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
        }, epoch + 1)

    logger.close()
    am_probes.close()
    am_probes_ema.close()


def geometric_mean(accperclass: np.ndarray) -> float:
    gm = 1.0
    for i in range(num_class):
        if accperclass[i] == 0:
            gm *= (1 / (100 * num_class)) ** (1 / num_class)
        else:
            gm *= (accperclass[i]) ** (1 / num_class)
    return float(gm)


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion,
          epoch: int, am_probes: ActivationMatchedProbes, global_iter: int):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()

    # logs
    probe_ready_m = AverageMeter()
    bias_norm_m = AverageMeter()

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

        # Periodically update probes to match current BN stats (warmup before using them)
        am_probes.maybe_update(model, H, W, steps=int(args.am_steps))

        ready = float(am_probes.global_iter >= int(args.am_warmup_iters))
        probe_ready_m.update(ready, 1)

        with torch.no_grad():
            outputs_u, _ = model(inputs_u)
            if epoch > args.debiasstart and ready > 0.5:
                biaseddegree = am_probes.bias_logits(model, H, W)
                bias_norm_m.update(biaseddegree.norm(p=2).item(), 1)
                outputs_u = outputs_u - biaseddegree.detach()

            targets_u2 = F.softmax(outputs_u, dim=1).detach()

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

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | ETA: {eta:} | ' \
                     'Loss: {loss:.4f} | Lx: {loss_x:.4f} | Lu: {loss_u:.4f} | ProbeReady: {pr:.2f}'.format(
                        batch=batch_idx + 1, size=args.val_iteration,
                        data=data_time.avg, bt=batch_time.avg, eta=bar.eta_td,
                        loss=losses.avg, loss_x=losses_x.avg, loss_u=losses_u.avg,
                        pr=probe_ready_m.avg
                     )
        if epoch > args.debiasstart and bias_norm_m.count > 0:
            bar.suffix += ' | ||b||: %.3f' % bias_norm_m.avg
        bar.next()

    bar.finish()
    return global_iter


def validate(valloader, model, criterion, mode: str, epoch: int, am_probes_ema: ActivationMatchedProbes):
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
        first_batch = next(iter(valloader))
        x0 = first_batch[0].cuda()
        H, W = int(x0.shape[2]), int(x0.shape[3])

    # Adapt EMA probes a little (optional) so they match EMA BN running stats
    if int(args.am_test_adapt_steps) > 0:
        with torch.enable_grad():
            am_probes_ema.maybe_update(model, H, W, steps=int(args.am_test_adapt_steps))

    with torch.no_grad():
        biaseddegree = am_probes_ema.bias_logits(model, H, W)

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
                batch=batch_idx + 1, size=len(valloader),
                data=data_time.avg, bt=batch_time.avg,
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
            param.mul_(1 - self.wd)


if __name__ == '__main__':
    main()
