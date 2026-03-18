# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP)
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math
import numpy as np
import csv
import contextlib

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


DIAG_FIELDS = [
    'phase', 'epoch', 'iter', 'lr',
    'loss', 'loss_x', 'loss_u',
    'accept_rate', 'bias_l2', 'bias_entropy',
    'align_loss', 'tv_loss', 'feat_center_dist', 'xbase_l2',
    'top1_raw', 'top1_debias', 'bacc_raw', 'bacc_debias', 'gm_raw', 'gm_debias'
]
DIAG_F = None
DIAG_WRITER = None


@contextlib.contextmanager
def temporary_eval(model: nn.Module):
    was_training = model.training
    try:
        model.eval()
        yield
    finally:
        if was_training:
            model.train()


def flatten_feature(feat: torch.Tensor) -> torch.Tensor:
    if feat.dim() == 4:
        return feat.view(feat.size(0), -1)
    if feat.dim() == 2:
        return feat
    if feat.dim() == 1:
        return feat.view(1, -1)
    return feat.reshape(feat.size(0), -1)


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
    return dh.abs().mean() + dw.abs().mean()


def entropy_from_logits(logits_1d: torch.Tensor) -> float:
    p = F.softmax(logits_1d, dim=-1)
    h = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return float(h.item())


def geometric_mean_from_acc(class_acc: np.ndarray, class_num: int) -> float:
    gm = 1.0
    for i in range(class_num):
        if class_acc[i] == 0:
            gm *= (1 / (100 * class_num)) ** (1 / class_num)
        else:
            gm *= (class_acc[i]) ** (1 / class_num)
    return gm


def diag_init(out_dir: str, filename: str):
    global DIAG_F, DIAG_WRITER
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    is_new = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    DIAG_F = open(path, 'a', newline='')
    DIAG_WRITER = csv.DictWriter(DIAG_F, fieldnames=DIAG_FIELDS)
    if is_new:
        DIAG_WRITER.writeheader()
        DIAG_F.flush()


def diag_write(row: dict):
    if DIAG_WRITER is None:
        return
    for k in DIAG_FIELDS:
        row.setdefault(k, '')
    DIAG_WRITER.writerow(row)
    DIAG_F.flush()


def diag_close():
    global DIAG_F, DIAG_WRITER
    if DIAG_F is not None:
        DIAG_F.flush()
        DIAG_F.close()
    DIAG_F = None
    DIAG_WRITER = None


class FMABaseState:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.feature_ema = None
        self.x_base = None
        self.optimizer = None
        self.last_align_loss = 0.0
        self.last_tv_loss = 0.0
        self.last_feat_dist = 0.0

    def init_from_batch(self, x_ref: torch.Tensor, lr: float):
        if self.x_base is not None:
            return
        x0 = x_ref.mean(dim=0, keepdim=True).detach().clone()
        self.x_base = nn.Parameter(x0)
        self.optimizer = optim.Adam([self.x_base], lr=lr)

    @torch.no_grad()
    def update_feature_center(self, feat_batch: torch.Tensor, alpha: float):
        f_mean = feat_batch.mean(dim=0)
        if self.feature_ema is None:
            self.feature_ema = f_mean.detach().clone()
        else:
            self.feature_ema.mul_(alpha).add_(f_mean * (1.0 - alpha))

    def optimize_xbase(self, model: nn.Module, steps: int, tv_weight: float, clamp_min: float, clamp_max: float):
        if self.x_base is None or self.feature_ema is None or self.optimizer is None:
            return

        param_requires_grad = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)

        with temporary_eval(model):
            with torch.enable_grad():
                for _ in range(max(1, steps)):
                    self.optimizer.zero_grad()
                    outputs = model(self.x_base, return_feature=True)
                    feat = flatten_feature(outputs[2])
                    center = self.feature_ema.detach().view(1, -1)
                    align = ((feat - center) ** 2).mean()
                    tv = tv_loss(self.x_base)
                    loss = align + tv_weight * tv
                    loss.backward()
                    self.optimizer.step()
                    with torch.no_grad():
                        self.x_base.clamp_(clamp_min, clamp_max)

                    self.last_align_loss = float(align.item())
                    self.last_tv_loss = float(tv.item())
                    self.last_feat_dist = float(torch.norm((feat - center).detach(), p=2).item())

        for p, flag in zip(model.parameters(), param_requires_grad):
            p.requires_grad_(flag)

    @torch.no_grad()
    def bias_logits(self, model: nn.Module) -> torch.Tensor:
        if self.x_base is None:
            return torch.zeros(self.num_classes, device='cuda')
        with temporary_eval(model):
            with torch.inference_mode():
                out = model(self.x_base)
                logits = out[0] if isinstance(out, (tuple, list)) else out
        return logits.view(-1).detach()


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

# dataset and imbalanced type
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset')
parser.add_argument('--imbalancetype', type=str, default='long', help='Long tailed or step imbalanced')
parser.add_argument('--unlabeledratio', type=float, default=2, help='Long tailed or step imbalanced')
parser.add_argument('--debiasstart', type=int, default=100, help='Long tailed or step imbalanced')

# FMA-Base options
parser.add_argument('--fma-enable', dest='fma_enable', action='store_true',
                    help='Enable Feature-Mean Anchored baseline picture')
parser.add_argument('--no-fma', dest='fma_enable', action='store_false',
                    help='Disable FMA and fallback to white baseline')
parser.set_defaults(fma_enable=True)
parser.add_argument('--fma-alpha', type=float, default=0.99,
                    help='EMA factor for global feature mean')
parser.add_argument('--fma-update-freq', type=int, default=50,
                    help='Optimize baseline image every N iterations')
parser.add_argument('--fma-inner-steps', type=int, default=3,
                    help='Inner-loop gradient steps for baseline image')
parser.add_argument('--fma-inner-lr', type=float, default=0.05,
                    help='Inner-loop lr for baseline image optimization')
parser.add_argument('--fma-tv-weight', type=float, default=1e-4,
                    help='TV regularization weight for baseline image')
parser.add_argument('--fma-clamp-min', type=float, default=-3.0,
                    help='Clamp min for baseline image')
parser.add_argument('--fma-clamp-max', type=float, default=3.0,
                    help='Clamp max for baseline image')

# diagnostics csv
parser.add_argument('--diag-enable', dest='diag_enable', action='store_true',
                    help='Enable diagnostics csv logging')
parser.add_argument('--no-diag', dest='diag_enable', action='store_false',
                    help='Disable diagnostics csv logging')
parser.set_defaults(diag_enable=True)
parser.add_argument('--diag-freq', type=int, default=50,
                    help='Diagnostics write frequency in iterations')
parser.add_argument('--diag-file', type=str, default='fmabase_diag.csv',
                    help='Diagnostics CSV file name under --out')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
if args.dataset == 'cifar10':
    import dataset.fix_cifar10 as dataset
    print(f'==> Preparing imbalanced CIFAR10')
    num_class = 10
elif args.dataset == 'cifar100':
    import dataset.fix_cifar100 as dataset
    print(f'==> Preparing imbalanced CIFAR100')
    num_class = 100
    args.wd = 0.08
elif args.dataset == 'stl10':
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
    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    if args.diag_enable:
        diag_init(args.out, args.diag_file)

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

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                          drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=int(args.unlabeledratio * args.batch_size), shuffle=True, num_workers=4, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=200, shuffle=False, num_workers=4)

    print("==> creating WRN-28-2")

    def create_model(ema=False):
        model = models.WRN(2, num_classes=num_class)
        model = model.cuda()

        params = list(model.parameters())
        if ema:
            for param in params:
                param.detach_()

        return model, params

    model, params = create_model()
    ema_model, _ = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in params) / 1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
    fma_state = FMABaseState(num_class)
    start_epoch = 0

    title = 'fixcdmad-fmabase-' + args.dataset
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        if 'fma_x_base' in checkpoint and checkpoint['fma_x_base'] is not None:
            x_loaded = checkpoint['fma_x_base'].cuda()
            fma_state.x_base = nn.Parameter(x_loaded)
            fma_state.optimizer = optim.Adam([fma_state.x_base], lr=args.fma_inner_lr)
        if 'fma_feature_ema' in checkpoint and checkpoint['fma_feature_ema'] is not None:
            fma_state.feature_ema = checkpoint['fma_feature_ema'].cuda()

        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names([
            'Train Loss', 'Train Loss X', 'Train Loss U',
            'Accept Rate', 'Bias L2', 'Align Loss',
            'Top1 Raw', 'Top1 Debias', 'bACC Raw', 'bACC Debias', 'GM Raw', 'GM Debias'
        ])

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_stats = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, train_criterion, epoch, fma_state)
        test_acc_raw, testclassacc_raw, test_acc_deb, testclassacc_deb = validate(test_loader, ema_model, criterion, mode='Test Stats ', fma_state=fma_state)

        gm_raw = geometric_mean_from_acc(testclassacc_raw, num_class)
        gm_deb = geometric_mean_from_acc(testclassacc_deb, num_class)
        bacc_raw = float(testclassacc_raw.mean())
        bacc_deb = float(testclassacc_deb.mean())

        print('raw: top1:', test_acc_raw, 'bACC:', bacc_raw, 'GM:', gm_raw,
              '| debias: top1:', test_acc_deb, 'bACC:', bacc_deb, 'GM:', gm_deb)

        logger.append([
            train_stats['loss'], train_stats['loss_x'], train_stats['loss_u'],
            train_stats['accept_rate'], train_stats['bias_l2'], train_stats['align_loss'],
            test_acc_raw, test_acc_deb, bacc_raw, bacc_deb, gm_raw, gm_deb
        ])

        if args.diag_enable:
            diag_write({
                'phase': 'eval',
                'epoch': epoch,
                'iter': -1,
                'lr': optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else 0.0,
                'loss': train_stats['loss'],
                'loss_x': train_stats['loss_x'],
                'loss_u': train_stats['loss_u'],
                'accept_rate': train_stats['accept_rate'],
                'bias_l2': train_stats['bias_l2'],
                'bias_entropy': train_stats['bias_entropy'],
                'align_loss': train_stats['align_loss'],
                'tv_loss': train_stats['tv_loss'],
                'feat_center_dist': train_stats['feat_center_dist'],
                'xbase_l2': train_stats['xbase_l2'],
                'top1_raw': test_acc_raw,
                'top1_debias': test_acc_deb,
                'bacc_raw': bacc_raw,
                'bacc_debias': bacc_deb,
                'gm_raw': gm_raw,
                'gm_debias': gm_deb
            })

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'fma_x_base': fma_state.x_base.detach().cpu() if fma_state.x_base is not None else None,
                'fma_feature_ema': fma_state.feature_ema.detach().cpu() if fma_state.feature_ema is not None else None,
            }, epoch + 1)

    logger.close()
    if args.diag_enable:
        diag_close()


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, fma_state: FMABaseState):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    accept_meter = AverageMeter()
    bias_l2_meter = AverageMeter()
    bias_entropy_meter = AverageMeter()
    align_meter = AverageMeter()
    tv_meter = AverageMeter()
    feat_dist_meter = AverageMeter()
    xbase_l2_meter = AverageMeter()
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
            inputs_x, targets_x, _ = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = next(unlabeled_train_iter)

        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        targets_x2 = torch.zeros(batch_size, num_class).scatter_(1, targets_x.view(-1, 1), 1)

        inputs_x, targets_x2 = inputs_x.cuda(), targets_x2.cuda(non_blocking=True)
        inputs_u, inputs_u2, inputs_u3 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()

        if args.fma_enable and fma_state.x_base is None:
            fma_state.init_from_batch(inputs_u, args.fma_inner_lr)

        with torch.no_grad():
            outputs_u_full = model(inputs_u, return_feature=True)
            outputs_u = outputs_u_full[0]
            feat_u = flatten_feature(outputs_u_full[2])

            if args.fma_enable:
                fma_state.update_feature_center(feat_u, args.fma_alpha)

            if args.fma_enable and (epoch >= args.debiasstart) and (batch_idx % max(1, args.fma_update_freq) == 0):
                fma_state.optimize_xbase(model, args.fma_inner_steps, args.fma_tv_weight, args.fma_clamp_min, args.fma_clamp_max)

            if epoch > args.debiasstart:
                if args.fma_enable and fma_state.x_base is not None:
                    biaseddegree = fma_state.bias_logits(model)
                else:
                    white = torch.ones((1, inputs_u.size(1), inputs_u.size(2), inputs_u.size(3)), device=inputs_u.device)
                    with temporary_eval(model):
                        with torch.inference_mode():
                            out = model(white)
                            biaseddegree = out[0].view(-1) if isinstance(out, (tuple, list)) else out.view(-1)
                outputs_u = outputs_u - biaseddegree.view(1, -1)
            else:
                biaseddegree = torch.zeros(num_class, device=inputs_u.device)

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

        accept_rate = float(max_p.ge(args.tau).float().mean().item())
        bias_l2 = float(torch.norm(biaseddegree.detach(), p=2).item())
        bias_entropy = entropy_from_logits(biaseddegree.detach())
        xbase_l2 = float(torch.norm(fma_state.x_base.detach(), p=2).item()) if (fma_state.x_base is not None) else 0.0

        accept_meter.update(accept_rate, 1)
        bias_l2_meter.update(bias_l2, 1)
        bias_entropy_meter.update(bias_entropy, 1)
        align_meter.update(fma_state.last_align_loss, 1)
        tv_meter.update(fma_state.last_tv_loss, 1)
        feat_dist_meter.update(fma_state.last_feat_dist, 1)
        xbase_l2_meter.update(xbase_l2, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        if args.diag_enable and (batch_idx % max(1, args.diag_freq) == 0):
            diag_write({
                'phase': 'train',
                'epoch': epoch,
                'iter': batch_idx,
                'lr': optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else 0.0,
                'loss': float(loss.item()),
                'loss_x': float(Lx.item()),
                'loss_u': float(Lu.item()),
                'accept_rate': accept_rate,
                'bias_l2': bias_l2,
                'bias_entropy': bias_entropy,
                'align_loss': fma_state.last_align_loss,
                'tv_loss': fma_state.last_tv_loss,
                'feat_center_dist': fma_state.last_feat_dist,
                'xbase_l2': xbase_l2
            })

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                     'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | Accpt: {accpt:.3f} | bL2: {bl2:.3f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    accpt=accept_meter.avg,
                    bl2=bias_l2_meter.avg,
                    )
        bar.next()
    bar.finish()

    return {
        'loss': losses.avg,
        'loss_x': losses_x.avg,
        'loss_u': losses_u.avg,
        'accept_rate': accept_meter.avg,
        'bias_l2': bias_l2_meter.avg,
        'bias_entropy': bias_entropy_meter.avg,
        'align_loss': align_meter.avg,
        'tv_loss': tv_meter.avg,
        'feat_center_dist': feat_dist_meter.avg,
        'xbase_l2': xbase_l2_meter.avg
    }


def validate(valloader, model, criterion, mode, fma_state: FMABaseState):
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
        if args.fma_enable and fma_state.x_base is not None:
            biaseddegree = fma_state.bias_logits(model).view(1, -1)
        else:
            first_batch = next(iter(valloader))
            x_ref = first_batch[0].cuda(non_blocking=True)
            white = torch.ones((1, x_ref.size(1), x_ref.size(2), x_ref.size(3)), device=x_ref.device)
            with temporary_eval(model):
                with torch.inference_mode():
                    out = model(white)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
            biaseddegree = logits.view(1, -1)

        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            targetsonehot = torch.zeros(inputs.size()[0], num_class).scatter_(1, targets.cpu().view(-1, 1).long(), 1)
            outputs, _ = model(inputs)
            outputs2 = outputs - biaseddegree

            score = F.softmax(outputs, dim=1)
            score2 = F.softmax(outputs2, dim=1)

            prediction = torch.argmax(score, dim=1)
            prediction2 = torch.argmax(score2, dim=1)

            outputs2onehot = torch.zeros(inputs.size()[0], num_class).scatter_(1, prediction.cpu().view(-1, 1).long(), 1)
            outputs2onehot2 = torch.zeros(inputs.size()[0], num_class).scatter_(1, prediction2.cpu().view(-1, 1).long(), 1)

            accperclass = accperclass + torch.sum(targetsonehot * outputs2onehot, dim=0).cpu().detach().numpy().astype(np.int64)
            accperclass2 = accperclass2 + torch.sum(targetsonehot * outputs2onehot2, dim=0).cpu().detach().numpy().astype(np.int64)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            prec1debias, prec5debias = accuracy(outputs2, targets, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            top1debias.update(prec1debias.item(), inputs.size(0))
            top5debias.update(prec5debias.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
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

    if args.dataset == 'cifar10':
        accperclass = accperclass / 1000
        accperclass2 = accperclass2 / 1000
    elif args.dataset == 'stl10':
        accperclass = accperclass / 800
        accperclass2 = accperclass2 / 800
    elif args.dataset == 'cifar100':
        accperclass = accperclass / 100
        accperclass2 = accperclass2 / 100

    return top1.avg, accperclass, top1debias.avg, accperclass2


def f(x, a, b, c, d):
    return np.sum(a * b * np.exp(-1 * x / c)) - d


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
    if imb == 'step':
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
    main()
