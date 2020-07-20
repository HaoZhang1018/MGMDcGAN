from __future__ import print_function

import time

# from utils import list_images
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from train import train
from generate import generate
import scipy.ndimage

BATCH_SIZE = 24
EPOCHES = 2
LOGGING = 20
MODEL_SAVE_PATH = './models/'
IS_TRAINING = True
f = h5py.File('Medical_dataset.h5', 'r')
for key in f.keys():
	print(f[key].name)
sources = f['data'][:]

sources = np.transpose(sources, (0, 3, 2, 1))
print('source shape:', sources.shape)


# for i in range(int(sources.shape[0])):
# 	ir_ds = scipy.ndimage.zoom(sources[i, :, :, 1], 0.25)
# 	ir_ds_us = scipy.ndimage.zoom(ir_ds, 4, order = 3)
# 	sources[i, :, :, 1] = ir_ds_us
#
# if not os.path.exists('Dataset3_ds_us.h5'):
# 	with h5py.File('Dataset3_ds_us.h5') as f2:
# 		f2['data'] = sources

def main():
	if IS_TRAINING:
		print(('\nBegin to train the network ...\n'))
		train(sources, MODEL_SAVE_PATH, EPOCHES, BATCH_SIZE, logging_period = LOGGING)
	else:
		print('\nBegin to generate pictures ...\n')
		T1path = './medical_test_imgs/MR-T1/'
		T2path = './medical_test_imgs/MR-T2/'
		PETpath = './medical_test_imgs/PET-I/'

		# for root, dirs, files in os.walk(path):
		# 	test_num = len(files)

		# for k in range(1):
		# 	model_num = 10 * (k + 75)
		t = []
		# Except = [5, 6, 13, 17, 19]
		index = 960
		for i in range(20):
			num = 3 * (i + 14)
			MODEL_PATH = MODEL_SAVE_PATH + str(index) + '/' + str(index) + '.ckpt'
			savepath = './medical_fused_imgs/'
			t1_path = T1path + '0' + str(num) + '.png'
			t2_path = T2path + '0' + str(num) + '.png'
			pet_path = PETpath + '0' + str(num) + '.png'

			# generate(ir_path, vis_path, MODEL_SAVE_PATH + str(model_num) + '/' + str(model_num) + '.ckpt', index,
			#          model_num = model_num, output_path = savepath)
			T=generate(pet_path, t1_path, MODEL_PATH, num, model_num = index, output_path = savepath)
			t.append(T)
			print("model_num:%s, pic_num:%s" % (index, num))
			print("mean:%s, std: %s" % (np.mean(t), np.std(t)))


# end = time.time()
# t.append(end - begin)


# print("mean:%s, std: %s" % (np.mean(t), np.std(t)))


if __name__ == '__main__':
	main()
