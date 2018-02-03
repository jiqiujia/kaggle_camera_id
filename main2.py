# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time
import numpy as np
import jpeg4py as jpeg
from PIL import Image
from io import BytesIO
import cv2
import math
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import  islice
import glob
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
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
parser.add_argument('--extra_dataset', '--ed', action='store_true')

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
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if CUDA:
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

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
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if CUDA:
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        bs, ncrops, c, h, w = input.size()
        input_var = input_var.view(-1, c, h, w)
        # compute output
        output = model(input_var)
        output = output.view(bs, ncrops, -1).mean(1)
        loss = criterion(output, target_var)

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
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
CLASSES = [
    'HTC-1-M7',
    'iPhone-6',     
    'Motorola-Droid-Maxx',
    'Motorola-X',
    'Samsung-Galaxy-S4',
    'iPhone-4s',
    'LG-Nexus-5x', 
    'Motorola-Nexus-6',
    'Samsung-Galaxy-Note3',
    'Sony-NEX-7']

EXTRA_CLASSES = [
    'htc_m7',
    'iphone_6',
    'moto_maxx',
    'moto_x',
    'samsung_s4',
    'iphone_4s',
    'nexus_5x',
    'nexus_6',
    'samsung_note3',
    'sony_nex7'
]
RESOLUTIONS = {
    0: [[1520,2688]], # flips
    1: [[3264,2448]], # no flips
    2: [[2432,4320]], # flips
    3: [[3120,4160]], # flips
    4: [[4128,2322]], # no flips
    5: [[3264,2448]], # no flips
    6: [[3024,4032]], # flips
    7: [[1040,780],  # Motorola-Nexus-6 no flips
        [3088,4130], [3120,4160]], # Motorola-Nexus-6 flips
    8: [[4128,2322]], # no flips 
    9: [[6000,4000]], # no flips
}
N_CLASSES = len(CLASSES)
load_img_fast_jpg  = lambda img_path: jpeg.JPEG(img_path).decode()
load_img  = lambda img_path: np.array(Image.open(img_path))

CROP_SIZE = 256
SEED=111
EXTRA_TRAIN_FOLDER = 'flickr_images'
EXTRA_VAL_FOLDER   = 'val_images'
TEST_FOLDER        = 'test/test'
MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0', 'Nothing']

def random_manipulation(img, manipulation=None):

    if manipulation == None:
        manipulation = np.random.choice(MANIPULATIONS)

    if manipulation.startswith('jpg'):
        quality = int(manipulation[3:])
        out = BytesIO()
        im = Image.fromarray(img)
        im.save(out, format='jpeg', quality=quality)
        im_decoded = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()
        del out
        del im
    elif manipulation.startswith('gamma'):
        gamma = float(manipulation[5:])
        # alternatively use skimage.exposure.adjust_gamma
        # img = skimage.exposure.adjust_gamma(img, gamma)
        im_decoded = np.uint8(cv2.pow(img / 255., gamma)*255.)
    elif manipulation.startswith('bicubic'):
        scale = float(manipulation[7:])
        im_decoded = cv2.resize(img,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    else:
        return img
    return im_decoded

def get_crop(img, crop_size, random_crop=False):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    half_crop = crop_size // 2
    pad_x = max(0, crop_size - img.shape[1])
    pad_y = max(0, crop_size - img.shape[0])
    if (pad_x > 0) or (pad_y > 0):
        img = np.pad(img, ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2), (0,0)), mode='wrap')
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    if random_crop:
        freedom_x, freedom_y = img.shape[1] - crop_size, img.shape[0] - crop_size
        if freedom_x > 0:
            center_x += np.random.randint(math.ceil(-freedom_x/2), freedom_x - math.floor(freedom_x/2) )
        if freedom_y > 0:
            center_y += np.random.randint(math.ceil(-freedom_y/2), freedom_y - math.floor(freedom_y/2) )

    return img[center_y - half_crop : center_y + crop_size - half_crop, 
               center_x - half_crop : center_x + crop_size - half_crop]

def get_class(class_name):
    if class_name in CLASSES:
        class_idx = CLASSES.index(class_name)
    elif class_name in EXTRA_CLASSES:
        class_idx = EXTRA_CLASSES.index(class_name)
    else:
        assert False
    assert class_idx in range(N_CLASSES)
    return class_idx
    
def process_item(item, training, transforms=[[]]):

    class_name = item.split('/')[-2]
    class_idx = get_class(class_name)

    img = load_img_fast_jpg(item)

    shape = list(img.shape[:2])

    # discard images that do not have right resolution
    if shape not in RESOLUTIONS[class_idx]:
        return None

    # some images may not be downloaded correclty and are B/W, discard those
    if img.ndim != 3:
        return None

    if len(transforms) == 1:
        _img = img
    else:
        _img = np.copy(img)

        img_s         = [ ]
        manipulated_s = [ ]
        class_idx_s   = [ ]

    for transform in transforms:

        force_manipulation = 'manipulation' in transform
        force_orientation  = 'orientation'  in transform

        # some images are landscape, others are portrait, so augment training by randomly changing orientation
        if ((np.random.rand() < 0.5) and training) or force_orientation:
            img = np.swapaxes(_img, 0,1)
        else:
            img = _img

        img = get_crop(img, CROP_SIZE * 2, random_crop=True if training else False) # * 2 bc may need to scale by 0.5x and still get a 512px crop

        if args.verbose:
            print("om: ", img.shape, item)

        manipulated = 0.
        if ((np.random.rand() < 0.5) and training) or force_manipulation:
            img = random_manipulation(img)
            manipulated = 1.
            if args.verbose:
                print("am: ", img.shape, item)

        img = get_crop(img, CROP_SIZE, random_crop=True if training else False)
        if args.verbose:
            print("ac: ", img.shape, item)

        img = torchvision.transforms.ToTensor(img)
        img = utils.normalize(img)
        if args.verbose:
            print("ap: ", img.shape, item)

        if len(transforms) > 1:
            img_s.append(img)    
            manipulated_s.append(manipulated)
            class_idx_s.append(class_idx)

    if len(transforms) == 1:
        return img, manipulated, class_idx
    else:
        return img_s, manipulated_s, class_idx_s

VALIDATION_TRANSFORMS = [ [], ['orientation'], ['manipulation'], ['orientation','manipulation']]

def gen(items, batch_size, training=True, inference=False):

    validation = not training 

    # during validation we store the unaltered images on batch_idx and a manip one on batch_idx + batch_size, hence the 2
    valid_batch_factor = 1 # TODO: augment validation

    # X holds image crops
    X = np.empty((batch_size * valid_batch_factor, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)
    # O whether the image has been manipulated (1.) or not (0.)
    O = np.empty((batch_size * valid_batch_factor, 1), dtype=np.float32)

    # class index
    y = np.empty((batch_size * valid_batch_factor), dtype=np.int64)
    
    p = Pool(cpu_count()-2)

    transforms = VALIDATION_TRANSFORMS if validation else [[]]

    assert batch_size % len(transforms) == 0

    while True:

        if training:
            np.random.shuffle(items)

        process_item_func  = partial(process_item, training=training, transforms=transforms)

        batch_idx = 0
        iter_items = iter(items)
        for item_batch in iter(lambda:list(islice(iter_items, batch_size)), []):

            batch_results = p.map(process_item_func, item_batch)
            for batch_result in batch_results:

                if batch_result is not None:
                    if len(transforms) == 1:
                        X[batch_idx], O[batch_idx], y[batch_idx] = batch_result
                        batch_idx += 1
                    else:
                        for _X,_O,_y in zip(*batch_result):
                            X[batch_idx], O[batch_idx], y[batch_idx] = _X,_O,_y
                            batch_idx += 1

                if batch_idx == batch_size:
                    yield([X, O], [y])
                    batch_idx = 0
                    
ids = glob.glob(os.path.join(args.train_dir,'*/*.jpg'))
ids.sort()

if not args.extra_dataset:
    ids_train, ids_val = train_test_split(ids, test_size=0.1, random_state=SEED)
else:
    ids_train = ids
    ids_val   = [ ]

    extra_train_ids = [os.path.join(EXTRA_TRAIN_FOLDER,line.rstrip('\n')) for line in open(os.path.join(EXTRA_TRAIN_FOLDER, 'good_jpgs'))]
    extra_train_ids.sort()
    ids_train.extend(extra_train_ids)

    extra_val_ids = glob.glob(os.path.join(EXTRA_VAL_FOLDER,'*/*.jpg'))
    extra_val_ids.sort()
    ids_val.extend(extra_val_ids)

classes = [get_class(idx.split('/')[-2]) for idx in ids_train]

classes_count = np.bincount(classes)
for class_name, class_count in zip(CLASSES, classes_count):
    print('{:>22}: {:5d} ({:04.1f}%)'.format(class_name, class_count, 100. * class_count / len(classes)))

class_weight = class_weight.compute_class_weight('balanced', np.unique(classes), classes)

ids_test = glob.glob(os.path.join(TEST_FOLDER,'*.tif'))

train_loader = gen(ids_train, args.batch_size)
val_loader = gen(ids_val, args.batch_size)
test_loader = gen(ids_test, args.batch_size)

#%%
print(args)
#original_model = models.resnet101(pretrained=True)
original_model = globals()[args.arch](pretrained=args.pretrained)
if args.finetune:
    for param in original_model.parameters():
        param.requires_grad = False

if args.arch.startswith('densenet'):
    original_model.classifier = nn.Linear(1024, 10)
    model = original_model
else:
    original_model.fc = nn.Linear(2048, 10)
    model = original_model
    #model = utils.FineTuneModel(original_model, args.arch, 10)
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
for epoch in range(args.num_epochs):
    if not args.adam:
        adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
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
df.to_csv('predict.csv', index=False)
#if os.path.isfile(args.resume):
#    print("=> loading checkpoint '{}'".format(args.resume))
#    checkpoint = torch.load(args.resume)
#    args.start_epoch = checkpoint['epoch']
#    best_prec1 = checkpoint['best_prec1']
#    model.load_state_dict(checkpoint['state_dict'])
#    print("=> loaded checkpoint '{}' (epoch {})"
#          .format(args.evaluate, checkpoint['epoch']))
#else:
#    print("=> no checkpoint found at '{}'".format(args.resume))
