# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
#import jpeg4py as jpeg
import PIL
import functools
import cv2
import math
from sklearn.utils import class_weight
from myfolder import MyImageFolder

import torch
import torch.utils.data.sampler as sampler
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models import *

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

def random_crop_fft(img, crop_size):
    nr, nc = img.shape[:2]
    r1, c1 = np.random.randint(nr-crop_size), np.random.randint(nc-crop_size) 
    img = img[r1:r1+crop_size, c1:c1+crop_size, :]
    img -= cv2.GaussianBlur(img, (3,3), 0)
    sf = np.stack([np.abs( np.fft.fftshift(np.fft.fft2(img[:,:,c])) ) for c in range(3)], axis=-1)
    return np.abs(sf)

def imread_residual_fft(img, crop_size, navg=64):
    #print(fn, rss())
    img = np.array(img).astype(np.float32) / 255.0
    return functools.reduce(lambda x,y:x+y, map(lambda x:random_crop_fft(img, crop_size), range(navg))) / navg
    #return random_crop_fft(img, crop_size)
   # with Pool() as pool:
   #     s = functools.reduce(lambda x,y:x+y, 
   #             tqdm.tqdm(pool.imap(random_crop_fft, itertools.repeat(img, navg)), 
   #                 total=navg)) / navg
   #             return s
 
def fft(img, crop_size, num_channels=3, **kwargs):
    if num_channels==3:
     #   return img
        return transforms.RandomCrop(crop_size)(img)
    im_fft = imread_residual_fft(img, crop_size)

    #img = np.array(img).astype(np.float32)/255.0
    #im_fft = img - cv2.GaussianBlur(img, (3,3), 0)
    #im_fft = np.abs(np.fft.fftshift(np.fft.fft2(img[:,:,0])))
    #for chan in range(3):
    #    tmp = im_fft[:,:,chan]
    #    mean_v = 2*np.median(tmp[:])
    #    tmp[tmp>=mean_v] = 0.0
    #    im_fft[:,:,chan] = tmp
    im_fft = im_fft[:,:,0]
    median_v = 2*np.median(im_fft)
    im_fft[im_fft>median_v]=0.0
    im_fft = (im_fft * 255. / np.max(im_fft)).astype(np.uint8)
    im_fft = im_fft.reshape((im_fft.shape[0], im_fft.shape[1], 1))
    #im_fft = np.stack([im_fft[:,:,c] * 255 / np.max(im_fft[:,:,c]) for c in range(3)], axis=-1).astype(np.uint8)
    if num_channels>3:
        #im = transforms.RandomCrop(crop_size)(im)
        img = np.array(img)
        img = np.concatenate([img, im_fft], axis=2)
        return img
    else:
        return im_fft

def plot_images(images, cls_true, cls_pred=None):

    #assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i, :, :, :], interpolation='spline16')
        # get its equivalent class name
        cls_true_name = cls_true[i]
            
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = cls_pred[i]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
            
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    
def resize(img, r=None):
    if r is None:
        r = np.random.uniform(0.5, 2.0)
    w, h = img.size
    w = int(w*r)
    h = int(h*r)
    return img.resize([w,h], Image.BICUBIC)

def gamma(img, g=None):
    if g is None:
        g = np.random.uniform(0.8, 1.2)
    
    invGamma = 1.0/g
    table = np.array([((i / 255.0) ** invGamma) * 255
                for i in np.arange(0, 256)]).astype("uint8")
    img = img.point(lambda x: table[x])
    return img

def high_pass_filter(im, filtering=False):
    if not filtering:
        return im
    kernel = [-1, 2, -2, 2, -1, 
            2, -6, 8, -6, 2,
            -2, 8, -12, 8, -2,
            2, -6, 8, -6, 2,
            -1, 2, -2, 2, -1]
    kernel=PIL.ImageFilter.Kernel((5,5), kernel, 1.0/12)
    im = im.filter(kernel)
    return im

MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0', 'Nothing']
#MANIPULATIONS = ['jpg70', 'jpg90', 'gamma', 'bicubic']
def manipulate(img):
    manipulation = np.random.choice(MANIPULATIONS)
    if manipulation.startswith('jpg'):
        quality = int(manipulation[3:])
        out = BytesIO()
        #im = Image.fromarray(img)
        img.save(out, format='jpeg', quality=quality)
        #im_decoded = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()
        img = Image.open(out)
        #im_decoded = Image.fromarray(im_decoded)
    elif manipulation.startswith('gamma'):
        g= float(manipulation[5:])
        img = gamma(img, g)
        # alternatively use skimage.exposure.adjust_gamma
        # img = skimage.exposure.adjust_gamma(img, gamma)
        #im_decoded = np.uint8(cv2.pow(img / 255., gamma)*255.)
    elif manipulation.startswith('bicubic'):
        scale = float(manipulation[7:])
        img = resize(img, scale)
        #im_decoded = cv2.resize(img,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    
    return img

def orient(img):
    r = np.random.randint(0,2)
    if r>0:
        img = img.rotate(-90, expand=1)
    return img

def get_train_valid_loader(data_dir,
                           batch_size,
                           crop_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           filtering=False,
                           num_channels=3,
                           shuffle=True,
                           show_sample=False,
                           num_workers=8,
                           pin_memory=False,
                           **kwargs):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # define transforms
    base_valid_transform = None
    base_train_transform = transforms.Compose([
        transforms.RandomCrop(crop_size*2 + 64)
    ])
    valid_transform = transforms.Compose([
        #transforms.Lambda(lambda img: high_pass_filter(img, filtering)),
        transforms.Lambda(lambda crop: manipulate(crop)),
        transforms.Lambda(lambda img: orient(img)),
        #transforms.RandomCrop(crop_size),
        transforms.Lambda(lambda crop: fft(crop, crop_size, num_channels, **kwargs)),
        #transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])

    
    if augment:
        train_transform = transforms.Compose([
            transforms.Lambda(lambda img: manipulate(img)),
            transforms.Lambda(lambda img: orient(img)),
            transforms.RandomHorizontalFlip(),
            #transforms.Lambda(lambda img: high_pass_filter(img, filtering)),
            #transforms.RandomCrop(crop_size),
            transforms.Lambda(lambda img: fft(img, crop_size, num_channels, **kwargs)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])

    # load the dataset
    if kwargs['l2_loss']:
        if not kwargs['same_crop']:
            train_transform = transforms.Compose([base_train_transform, train_transform])
            #valid_transform = transforms.Compose([base_valid_transform, valid_transform])
            #base_valid_transform = None
            base_train_transform = None
        train_transform1 = train_transform2 = train_transform
    else:
        train_transform1 = train_transform
        transform2 = None

    train_dataset = MyImageFolder(root=data_dir, transform=base_train_transform,
            transform1=train_transform, transform2 = train_transform)
    valid_dataset = MyImageFolder(root=data_dir, transform=base_valid_transform, 
            transform1=valid_transform, transform2 = valid_transform)

    extra_valid_dataset = MyImageFolder(root='./val_images', transform=base_valid_transform,
            transform1=valid_transform, transform2 = valid_transform)
    extra_valid_ys = np.array([img[1] for img in extra_valid_dataset.imgs])
    #extra_val_cws = class_weight.compute_class_weight('balanced', np.unique(extra_ys), extra_ys)
    num_extra_valid_imgs = len(extra_valid_dataset.imgs)
    extra_valid_cws = np.bincount(extra_valid_ys)#*1.0/num_extra_valid_imgs
    print('extra_valid_dataset size:', num_extra_valid_imgs)
    print('extra_valid_cws:' + ','.join(str(y) for y in extra_valid_cws))

    valid_dataset = torch.utils.data.ConcatDataset([extra_valid_dataset, valid_dataset]) 

    num_train = len(train_dataset)
    indices = np.arange(num_train)
    split = int(np.floor(valid_size * num_train))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx = indices[split:]
    valid_idx = np.concatenate((np.arange(len(extra_valid_dataset)), indices[:split] + len(extra_valid_dataset)))

    train_sampler = sampler.SubsetRandomSampler(train_idx)
    valid_sampler = sampler.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=batch_size, sampler=train_sampler, 
                    num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                    batch_size=batch_size, sampler=valid_sampler, 
                    num_workers=num_workers, pin_memory=pin_memory)


    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=9, 
                                                    shuffle=shuffle, 
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory)
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader, train_dataset)

def get_train_loader(data_dir, batch_size, crop_size, filtering=False, num_channels=False, shuffle=False, num_workers=8, pin_memory=False, **kwargs):
    base_transform = transforms.Compose([
        transforms.RandomCrop(crop_size*2 + 64)
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.Lambda(lambda img: manipulate(img)),
            transforms.Lambda(lambda img: orient(img)),
            transforms.RandomHorizontalFlip(),
            #transforms.Lambda(lambda img: high_pass_filter(img, filtering)),
            #transforms.RandomCrop(crop_size),
            transforms.Lambda(lambda img: fft(img, crop_size, num_channels, **kwargs)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])

    # load the dataset
    if kwargs['l2_loss']:
        transform1 = transform2 = train_transform
    else:
        transform1 = train_transform
        transform2 = None

    train_dataset = MyImageFolder(root=data_dir, transform=base_transform,
            transform1=train_transform, transform2 = train_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
            shuffle=shuffle, batch_size=batch_size, drop_last=True, 
            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, train_dataset


def get_val_loader(data_dir, 
                    batch_size,
                    crop_size,
                    filtering=False,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=False,
                    **kwargs):
    # define transform
    base_valid_transform = None
    valid_transform = transforms.Compose([
        #transforms.Lambda(lambda img: high_pass_filter(img, filtering)),
        transforms.Lambda(lambda crop: manipulate(crop)),
        transforms.Lambda(lambda img: orient(img)),
        #transforms.RandomCrop(crop_size),
        transforms.Lambda(lambda crop: fft(crop, crop_size, num_channels, **kwargs)),
        #transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])

    valid_dataset = MyImageFolder(root=data_dir, transform=base_valid_transform, 
            transform1=valid_transform, transform2 = valid_transform)
    
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=batch_size, 
                                              shuffle=shuffle, 
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)
    return data_loader, dataset
def get_test_loader(data_dir, 
                    batch_size,
                    crop_size,
                    filtering=False,
                    num_channels=3,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=False,
                    **kwargs):
    # define transform
    valid_transform = transforms.Compose([
        transforms.FiveCrop(crop_size),
        transforms.Lambda(lambda crops: [fft(crop, crop_size, num_channels, **kwargs) for crop in crops]),
        transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
        transforms.Lambda(lambda crops: torch.stack([
                normalize(crop) for crop in crops])),
    ])


    dataset = MyImageFolder(root=data_dir, transform=valid_transform)

    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=batch_size, 
                                              shuffle=shuffle, 
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)
    return data_loader, dataset


class TwoStreamModel(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, freeze=False):
        super(TwoStreamModel, self).__init__()

        model1 = globals()[arch](pretrained=pretrained)
        model2 = globals()[arch](pretrained=pretrained)
        self.rgb = FineTuneModel(model1, arch, num_classes).features
        self.noise = FineTuneModel(model2, arch, num_classes).features
        
        dim = list(model1.children())[-1].weight.shape[1]
        self.features = torch.cat([self.rgb, self.noise])
        self.classifier = nn.Linear(2*dim, num_classes)

    def forward(self, x):
        f = self.features(x)
        y = self.classifier(f)
        return y

class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes, num_channels=3, freeze=False, **kwargs):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet') :
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            dim = original_model.fc.weight.shape[1]
            print('dim:',dim)
            self.classifier = nn.Sequential(
                nn.Linear(dim, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('densenet'):
            self.features = original_model.features
            dim = original_model.classifier.weight.shape[1]
            print('dim:',dim)
            self.classifier = nn.Sequential(
                nn.Linear(dim, num_classes)
            )
            self.modelName = 'densenet'
        elif arch.startswith('vgg'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg'
        else :
            raise("Finetuning not supported on this architecture yet")

        # Freeze those weights
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
        elif num_channels!=3:
            pre_w = self.features[0].weight
            mean_w = torch.mean(pre_w, 1, keepdim=True)
            if num_channels==1:
                new_w = mean_w
            elif num_channels>3:
                mean_w = [mean_w] * (num_channels-3)
                new_w = torch.cat([pre_w] + mean_w, 1)
            else:
                assert False
            self.features = nn.Sequential(
                nn.Conv2d(num_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False),
                *list(self.features)[1:]
            )
            self.features.state_dict().__setitem__('0.weight', new_w)


    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'densenet':
            f = nn.AdaptiveAvgPool2d((1,1))(f)
        if self.modelName == 'alexnet' :
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg':
            f = f.view(f.size(0), -1)
        else:
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

class DualNet(nn.Module):
    def __init__(self, basenet):
        super(DualNet, self).__init__()
        self.features = basenet.features
        self.classifier = basenet.classifier
        self.modelName = basenet.modelName

    def forward(self, x1, x2):
        f1 = self.features(x1)
        f2 = self.features(x2)
        if self.modelName == 'densenet':
            f1 = nn.AdaptiveAvgPool2d((1,1))(f1)
            f2 = nn.AdaptiveAvgPool2d((1,1))(f2)
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        y1 = self.classifier(f1)
        y2 = self.classifier(f2)
        return f1, f2, y1, y2

def make_layers(cfg, in_channels, batch_norm=False):
    layers = []
    first = True
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v=='D':
            layers += [nn.Dropout(0.5)]
        else:
            if first:
                ks = 3
            else:
                ks = 3
                first = False
            conv2d = nn.Conv2d(in_channels, v, kernel_size=ks, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    layers += [nn.AdaptiveAvgPool2d((1,1))]
    return nn.Sequential(*layers)

class MyModel(nn.Module):
    def __init__(self, num_classes, num_kernels, num_channels=3, batch_norm=True, **kwargs):
        super(MyModel, self).__init__()
        self.cfg = {'A': [num_kernels, num_kernels, num_kernels, 'M', 'D',
            num_kernels*2, num_kernels*2, num_kernels*2, 'M', 'D',
            num_kernels*4, num_kernels*4, num_kernels*4, 'D'
            #num_kernels*8, num_kernels*8, 'M'
            ]
        }
        self.num_channels = num_channels
        
        self.features = make_layers(self.cfg['A'], self.num_channels, batch_norm=True)
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                #m.weight.data.normal_(0, 0.01)
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                m.bias.data.zero_()

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha>0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam*x + (1-lam)*x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam*criterion(pred, y_a) + (1-lam)*criterion(pred, y_b)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std
