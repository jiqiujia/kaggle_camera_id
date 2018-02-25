# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time
import numpy as np
from sklearn.metrics import confusion_matrix

import mymodels.resnet as myresnet
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models import *

import utils

ROOT = "./"
TRAIN_DIR = ROOT + "train3"
VAL_DIR = ROOT + 'val_images'
TEST_DIR = ROOT + "test"
BATCH_SIZE = 64
LR = 0.001
MOMENTUM = 0.9
WEIGTH_DECAY = 0.0005
ARCH = 'resnet'
NUM_EPOCHS = 100
CUDA = True

parser = argparse.ArgumentParser(description='PyTorch Example')
#parser.add_argument('--disable-cuda', action='store_true',
#                    help='Disable CUDA')
parser.add_argument('--train_dir', default='./train3', type=str)
parser.add_argument('--learning_rate', '--lr', default=0.001, type=float)
parser.add_argument('--batch_size', '--bs', default=64, type=int)
parser.add_argument('--arch', '-a', default='resnet101', type=str)
parser.add_argument('--resume', '-r', default='', type=str)
parser.add_argument('--finetune', '-f', default=0, type=int,
                    help='whether only finetune fc layers')
parser.add_argument('--pretrained', '-p', action='store_true')
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--num_epochs', '-e', default=0, type=int)
parser.add_argument('--step', '-s', default=30, type=int)
parser.add_argument('--filtering', action='store_true')
parser.add_argument('--adam', action='store_true')
parser.add_argument('--num_channels', '--nc', default=3, type=int)
parser.add_argument('--l2_loss', '--l2', action='store_true')
parser.add_argument('--crop_size', '--cs', default=224, type=int)
parser.add_argument('--valid_size', '--vs', default=0.1, type=float)
parser.add_argument('--same_crop', '--sc', action='store_true',
                    help='whether use the same crop to evaluate l2_loss')
parser.add_argument('--mixup', action='store_true')
parser.add_argument('--save_prefix', '--sp', default='', type=str)
parser.add_argument('--l2_loss_w', '--l2w', default=0, type=float)
parser.add_argument('--num_workers', '--nw', default=4, type=int)
parser.add_argument('--num_kernels', '--nk', default=64, type=int)
parser.add_argument('--augment', action='store_false')
parser.add_argument('--val_step', '--vsp', default=1, type=int)

args = parser.parse_args()
CUDA = CUDA and torch.cuda.is_available()
if CUDA:
    print("using gpu...")
else:
    print("using cpu...")

cudnn.benchmark = True

#from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
IMG_EXTENSIONS.append('.tif')
#ImageFolder('/raid/data/data1/kaggle_camera/test/')
#%%
l2_loss_w = args.l2_loss_w
print('l2_loss_w:', l2_loss_w)

save_prefix = '_'.join([args.arch, 'e'+str(args.num_epochs), 's'+str(args.step),
              'bs'+str(args.batch_size), 'lr'+str(args.learning_rate),
              'nc'+str(args.num_channels), 'vs'+str(args.valid_size)])
if args.adam:
    save_prefix += '_adam'
else:
    save_prefix += '_sgd'
if args.l2_loss:
    save_prefix += '_l2'+str(args.l2_loss_w)
save_prefix += '_'+ args.save_prefix
writer = SummaryWriter('runs/'+save_prefix)
def mse_loss(input, target):
    return torch.mean(torch.sum((input - target)**2, dim=1))

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    l2_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = data[0]
        target = data[-1]
        if args.l2_loss:
            dual_input = data[1]
            dual_input_var = torch.autograd.Variable(dual_input)  
        if CUDA:
            input = input.cuda(async=True)
            target = target.cuda(async=True)
        if args.mixup:
            input, y_a, y_b, lam = utils.mixup_data(input, target, alpha=1.0)
            y_a = torch.autograd.Variable(y_a)
            y_b = torch.autograd.Variable(y_b)


        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        if args.l2_loss:
            f1, f2, y1, y2 = model(input_var, dual_input_var)
            l2_loss = l2_loss_w * mse_loss(f1, f2)
            output = torch.cat([y1, y2])
            target = torch.cat([target, target])
            target_var = torch.cat([target_var, target_var])
            loss = criterion(output, target_var)
            loss = loss + l2_loss

            l2_losses.update(l2_loss.data[0], input.size(0))
        else:
            output = model(input_var)
            if args.mixup:
                loss_fun = utils.mixup_criterion(y_a, y_b, lam)
                loss = loss_fun(criterion, output)
            else:
                loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.mixup:
            _, predicted = torch.max(output.data, 1)
            prec1 = lam*predicted.eq(y_a.data).cpu().sum() + (1-lam)*predicted.eq(y_b.data).cpu().sum()
            top1.update(prec1, input.size(0))
        else:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1[0], input.size(0))
        #top5.update(prec5[0], input.size(0))

        losses.update(loss.data[0], input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 5 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'L2Loss {l2_loss.val:.4f} ({l2_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, l2_loss=l2_losses, top1=top1))
        
        step = epoch * len(train_loader) + i
        #print(type(step))
        writer.add_scalar('train/acc', prec1[0], step)
        writer.add_scalar('train/loss', loss.data[0], step)
        if args.l2_loss:
            writer.add_scalar('train/l2_loss', l2_loss.data[0], step)
        for name, param in model.named_parameters():
            #print(name, param.data.cpu().numpy().dtype)
            if name.find('batchnorm')==-1:
                writer.add_histogram(name, param.data.cpu().numpy(), step)

def validate(val_loader, model, criterion, multicrop=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    l2_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    outputs = []
    targets = []
    for i, data in enumerate(val_loader):
        input = data[0]
        target = data[-1]
        if args.l2_loss:
            dual_input = data[1]
            dual_input_var = torch.autograd.Variable(dual_input, volatile=True) 
        if CUDA:
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        if multicrop:
            bs, ncrops, c, h, w = input.size()
            input_var = input_var.view(-1, c, h, w)
        # compute output
        if args.l2_loss:
            if multicrop:
                dual_input_var = dual_input_var.view(-1, c, h, w)
            f1, f2, y1, y2 = model(input_var, dual_input_var)
            l2_loss = l2_loss_w * mse_loss(f1, f2)
            output = torch.cat([y1, y2])
            if multicrop:
                output = output.view(bs*2, ncrops, -1).mean(1)
            target = torch.cat([target, target])
            target_var = torch.cat([target_var, target_var])
            loss = criterion(output, target_var)
            loss = loss + l2_loss
            l2_losses.update(l2_loss.data[0], input.size(0))
        else:
            output = model(input_var)
            if multicrop:
                output = output.view(bs, ncrops, -1).mean(1)
            loss = criterion(output, target_var)

        outputs.append(output)
        targets.append(target)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 5 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'L2Loss {l2_loss.val:.4f} ({l2_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   l2_loss=l2_losses, top1=top1, top5=top5))
        step = epoch * len(val_loader) + i
        writer.add_scalar('val/acc', prec1[0], step)
        writer.add_scalar('val/loss', loss.data[0], step)
        if args.l2_loss:
            writer.add_scalar('val/l2_loss', l2_loss.data[0], step)

    outputs = torch.cat(outputs).data.cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()
    outputs = np.argmax(outputs, 1)
    #cm = confusion_matrix(outputs, targets)
    #print('confusion_matrix:', cm)
    print(' * Prec@1 {acc:.3f} Prec@5 {top5.avg:.3f}'
          .format(acc=np.mean(outputs==targets), top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = save_prefix + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_prefix + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.adam:
        if epoch<=args.step:
            lr = args.learning_rate
        else:
            lr = args.learning_rate * (1. - (epoch-args.step)/(args.num_epochs-args.step))
    else:
        lr = args.learning_rate * (0.1 ** (epoch // args.step))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr
                   
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
#%%
train_loader, val_loader, train_dataset = utils.get_train_valid_loader(args.train_dir,
        batch_size=args.batch_size, crop_size=args.crop_size,
        augment=args.augment, random_seed=111,
        shuffle=True, valid_size=args.valid_size,
        filtering=args.filtering, 
        num_channels=args.num_channels,
        l2_loss=args.l2_loss, same_crop=args.same_crop,
        num_workers=args.num_workers)
test_loader, test_dataset = utils.get_test_loader(TEST_DIR, batch_size=args.batch_size, crop_size=args.crop_size, filtering=args.filtering, num_channels=args.num_channels, l2_loss=args.l2_loss, num_workers=args.num_workers)
#val_loader, val_dataset = utils.get_val_loader(VAL_DIR, batch_size=args.batch_size, crop_size=args.crop_size, filtering=args.filtering, num_channels=args.num_channels, l2_loss=args.l2_loss)
print(train_dataset.classes)


print(args)
if args.arch.startswith('my'):
    model = myresnet.ResNet18()
else:
    #original_model = models.resnet101(pretrained=True)
    original_model = globals()[args.arch](pretrained=args.pretrained)
    if args.finetune:
        for param in original_model.parameters():
            param.requires_grad = False

    model = utils.FineTuneModel(original_model, args.arch, 10, num_channels=args.num_channels)

if args.l2_loss:
    model = utils.DualNet(model)

if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    model.features = torch.nn.DataParallel(model.features)
else:
    model = torch.nn.DataParallel(model)
if CUDA:
    model = model.cuda()

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
                .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

criterion = nn.CrossEntropyLoss()
if CUDA:
    criterion = criterion.cuda()

if args.adam:
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.learning_rate)
else:
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), # Only finetunable params
                                args.learning_rate, momentum=MOMENTUM, weight_decay=WEIGTH_DECAY)

print("begin training...")
print('using ' + args.train_dir)
best_prec1 = 0
#validate(val_loader, model, criterion)
for epoch in range(args.start_epoch, args.num_epochs):
    #if epoch==30:
    #    l2_loss_w = 0.1
    #if epoch==60:
    #    l2_loss_w = 1
    adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    if (epoch+1) % args.val_step==0:
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


#%% test
outputs = []
for i, (input, target) in enumerate(test_loader):
    target = target.cuda(async=True)
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)
    bs, ncrops, c, h, w = input.size()
    input_var = input_var.view(-1, c, h, w)

    # compute output
    if args.l2_loss:
        _, _, output, _ = model(input_var, input_var)
    else:
        output = model(input_var)
    output = output.view(bs, ncrops, -1).mean(1)
    outputs.append(output.data)
outputs = torch.cat(outputs)
outputs = outputs.cpu().numpy()
outputs = np.argmax(outputs, axis=1)
outputs = [train_dataset.classes[i] for i in outputs]

import pandas as pd
df = pd.DataFrame(columns=['fname', 'camera'])
df['fname'] = [img[0].split('/')[-1] for img in test_dataset.imgs]
df['camera'] = outputs
df.to_csv(save_prefix+'predict.csv', index=False)

writer.close()
