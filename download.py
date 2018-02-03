import os
import urllib

folder = '/media/dl/data1/datasets/kaggle_camera/moto_x_images/'
#folder = '/media/dl/data1/datasets/kaggle_camera/val_images/'
EXTRA_CLASSES = [
    #'htc_m7',
    #'iphone_6',
    #'moto_maxx',
    #'moto_x',
    #'samsung_s4',
    #'iphone_4s',
    #'nexus_5x',
    #'nexus_6',
    #'samsung_note3',
    'sony_nex7'
]

for cls in EXTRA_CLASSES:
	urls = os.path.join(folder, cls, 'urls_final')
	#urls = os.path.join(folder, cls, 'urls_dpreview')
	for url in open(urls):
		retry = 10
		while retry > 0:
			try:
				fname = url.rstrip('\n').split(os.sep)[-1]
				if os.path.exist(fname):
					break
				fname = os.path.join(folder, cls, fname)
				urllib.urlretrieve(url, fname)
				print('succesfully download ' + fname)
			except:
				print('error downloading ' + fname + str(retry))
				retry = retry - 1
				continue
			else:
				break
