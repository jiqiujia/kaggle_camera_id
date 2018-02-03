# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import glob, functools, tqdm, PIL
from multiprocessing import Pool
import itertools
from PIL import Image
from io import BytesIO


def imread(fn):
    return np.array(PIL.Image.open(fn))

def imsize(fn):
    im = PIL.Image.open(fn)
    sz = im.size
    im.close()
    return sz

def is_landscape(fn):
    s = imsize(fn)
    return s[0] > s[1]

train = pd.DataFrame({'path':glob.glob('../train/*/*')})
train['modelname'] = train.path.map(lambda p:p.split('/')[-2])

import cv2

def random_crop_fft(img, CROP_SIZE=256):
    nr, nc = img.shape[:2]
    r1, c1 = np.random.randint(nr-CROP_SIZE), np.random.randint(nc-CROP_SIZE) 
    img1 = img[r1:r1+CROP_SIZE, c1:c1+CROP_SIZE, :]
    #img1 -= cv2.GaussianBlur(img1, (3,3), 0)
    #img1 -= cv2.medianBlur(img1, 3)
    #img1 -= cv2.blur(img1, (3,3))
    for chan in range(3):
        img1[:,:,chan] -= cv2.fastNlMeansDenoising(img1[:,:,chan], None, 15, 5, 11)
    #img1 -= cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
    sf = np.stack([np.abs( np.fft.fftshift(np.fft.fft2(img1[:,:,c])) ) for c in range(3)], axis=-1)
    return np.abs(sf)
    
def imread_residual_fft(im, navg=16):
    #print(fn, rss())
    if type(im)==str:
        img = imread(im)
    else:
        img = np.array(im).astype(np.float32) / 255.0
    return functools.reduce(lambda x,y:x+y, map(lambda x:random_crop_fft(img), range(navg))) / navg
#    with Pool() as pool:
#        s = functools.reduce(lambda x,y:x+y, 
#                             tqdm.tqdm(pool.imap(random_crop_fft, itertools.repeat(img, navg)), 
#                                       total=navg)) / navg
#    return s
# 
def noise_pattern(modelname):
    files = train.path[train.modelname == modelname].values
    orientations = np.vectorize(is_landscape)(files)
    if np.sum(orientations) < len(orientations)//2:
        orientations = ~orientations
    files = files[orientations]

    s=imread_residual_fft(files[np.random.randint(0, 100)])
#    with Pool() as pool:
#        s = functools.reduce(lambda x,y:x+y, tqdm.tqdm(pool.imap(imread_residual_fft, files), total=len(files), desc=modelname)) / len(files)
#    
    return s
#%%
modelnames = train.modelname.unique()
noisepatterns = [noise_pattern(modelname) for modelname in modelnames]
for s in noisepatterns:
    for chan in range(3):
        tmp = s[:,:,chan]
        mean_v = 2*np.median(tmp[:])
        tmp[tmp>=mean_v] = 0.0
        s[:,:,chan] = tmp
#noisepatterns = [np.stack([s[:,:,chan] * 255 / np.max(s[:,:,chan]) for chan in range(3)], axis=-1).astype(np.uint8) for s in noisepatterns]
noisepatterns = [np.stack([(s[:,:,chan]-np.min(s[:,:,chan])) * 255 / (np.max(s[:,:,chan])-np.min(s[:,:,chan]))
 for chan in range(3)], axis=-1).astype(np.uint8) for s in noisepatterns]
#%%
#noisepatterns = [np.stack([s[:,:,chan]  / np.max(s[:,:,chan]) for chan in range(3)], axis=-1) for s in noisepatterns]
#_, ax = plt.subplots(3, 4, figsize=(16, 20))
#ax = ax.flatten()
#
#chan = 0
#for ax1, modelname, s in zip(ax[:len(modelnames)], modelnames, noisepatterns):
##    img = np.stack([cv2.equalizeHist(s[:,:,chan]) for chan in range(3)], axis=-1)
##    img = np.mean(img, axis=-1)
##    img = cv2.equalizeHist(s[:,:,chan])
#    img = s[:,:,chan]
#    ax1.imshow(img)
#    ax1.set_title(modelname)
#
#for ax1 in ax[len(modelnames):]:
#    ax1.axis('off')
#plt.show()
#%%
_, ax = plt.subplots(5, 4, figsize=(16, 20))
ax = ax.flatten()
num=0
for modelname, s in zip(modelnames, noisepatterns):
    img = cv2.equalizeHist(s[:,:,chan])
    print('s:', np.unique(s[:,:,chan][:]).shape, np.unique(s[:,:,chan][:]))
    print('img:', np.unique(img[:]).shape, np.unique(img[:]))
    ax[num].imshow(s)
    ax[num].set_title(modelname)
    num+=1
    ax[num].imshow(img)
    ax[num].set_title(modelname + '_hist')
    num+=1
for ax1 in ax[len(modelnames)*3:]:
    ax1.axis('off')
plt.show()

#%%
#im_name = '/media/dl/data1/datasets/kaggle_camera/test/test/img_0a3162a_unalt.tif'
#im_name = '/media/dl/data1/datasets/kaggle_camera/train/iPhone-4s/(iP4s)1.jpg'
#MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0', 'Nothing']
#def resize(img, r=None):
#    if r is None:
#        r = np.random.uniform(0.5, 2.0)
#    w, h = img.size
#    w = int(w*r)
#    h = int(h*r)
#    return img.resize([w,h], Image.BICUBIC)
#
#def gamma(img, g=None):
#    if g is None:
#        g = np.random.uniform(0.8, 1.2)
#    
#    invGamma = 1.0/g
#    table = np.array([((i / 255.0) ** invGamma) * 255
#                for i in np.arange(0, 256)]).astype("uint8")
#    img = img.point(lambda x: table[x])
#    return img
#def manipulate(img, manipulation):
#    if manipulation.startswith('jpg'):
#        quality = int(manipulation[3:])
#        out = BytesIO()
#        #im = Image.fromarray(img)
#        img.save(out, format='jpeg', quality=quality)
#        #im_decoded = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()
#        im_decoded = Image.open(out)
#        #im_decoded = Image.fromarray(im_decoded)
#        del out
#        del img
#    elif manipulation.startswith('gamma'):
#        g= float(manipulation[5:])
#        im_decoded = gamma(img, g)
#        # alternatively use skimage.exposure.adjust_gamma
#        # img = skimage.exposure.adjust_gamma(img, gamma)
#        #im_decoded = np.uint8(cv2.pow(img / 255., gamma)*255.)
#    elif manipulation.startswith('bicubic'):
#        scale = float(manipulation[7:])
#        im_decoded = resize(img, scale)
#        #im_decoded = cv2.resize(img,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
#    else:
#        return img
#    
#    return im_decoded
##%%
#_, ax = plt.subplots(4, 3, figsize=(16, 20))
#ax = ax.flatten()
#for ax1, man in zip(ax[:len(MANIPULATIONS)], MANIPULATIONS):
#    im = Image.open(im_name)
#    im = manipulate(im, man)
#    im_fft = imread_residual_fft(im)
#    im_fft_n = (np.abs(im_fft[:,:,1]) * 255 / np.max(np.abs(im_fft[:,:,1]))).astype(np.uint8)
#    im_fft_h = cv2.equalizeHist(im_fft_n)
#    ax1.imshow(im_fft_h)
#    ax1.set_title(man)
#for ax1 in ax[len(MANIPULATIONS):]:
#    ax1.axis('off')
#plt.show()
    