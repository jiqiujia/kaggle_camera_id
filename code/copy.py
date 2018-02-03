from shutil import copy2
from PIL import Image
import os

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
N_CLASSES = 10
def get_class(class_name):
    if class_name in CLASSES:
        class_idx = CLASSES.index(class_name)
    elif class_name in EXTRA_CLASSES:
        class_idx = EXTRA_CLASSES.index(class_name)
    else:
        assert False
    assert class_idx in range(N_CLASSES)
    return class_idx
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

for class_id,resolutions in RESOLUTIONS.copy().items():
    resolutions.extend([resolution[::-1] for resolution in resolutions])
    RESOLUTIONS[class_id] = resolutions


folder = '/media/dl/data1/datasets/kaggle_camera/flickr_images/'
for subdir, dirs, files in os.walk(folder):
	for file in files:
		if file[-3:] != 'jpg':
			continue
		cls = subdir.split(os.sep)[-1]
		cls_idx = get_class(cls)
		file_path = os.path.join(subdir, file)
		im = Image.open(file_path)
		shape = list(im.size)
		if shape in RESOLUTIONS[cls_idx]:
			copy2(file_path, os.path.join('/media/dl/data1/datasets/kaggle_camera/', 'train3', CLASSES[cls_idx], file))
			print(file_path)