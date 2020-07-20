# Train the DenseFuse Net

from __future__ import print_function

import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from scipy.misc import imsave
import scipy.ndimage

from Generator import Generator
from Discriminator import Discriminator1, Discriminator2
from LOSS import SSIM_LOSS, L1_LOSS, Fro_LOSS, _tf_fspecial_gauss
from generate import generate

patch_size = 84
# TRAINING_IMAGE_SHAPE = (patch_size, patch_size, 2)  # (height, width, color_channels)

LEARNING_RATE = 0.00015
EPSILON = 1e-5
DECAY_RATE = 0.9
eps = 1e-8

rc = 4


def train(source_imgs, save_path, EPOCHES_set, BATCH_SIZE, logging_period = 1):
	from datetime import datetime
	start_time = datetime.now()
	EPOCHS = EPOCHES_set
	print('Epoches: %d, Batch_size: %d' % (EPOCHS, BATCH_SIZE))

	MODEL_SAVE_PATH = save_path + 'temporary.ckpt'
	num_imgs = source_imgs.shape[0]
	mod = num_imgs % BATCH_SIZE
	n_batches = int(num_imgs // BATCH_SIZE)
	print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))

	if mod > 0:
		print('Train set has been trimmed %d samples...\n' % mod)
		source_imgs = source_imgs[:-mod]

	# create the graph
	with tf.Graph().as_default(), tf.Session() as sess:
		SOURCE_VIS = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'SOURCE_VIS')
		SOURCE_ir = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size / rc, patch_size / rc, 1),
		                           name = 'SOURCE_ir')
		print('source_vis shape:', SOURCE_VIS.shape)

		# upsampling vis and ir images
		G = Generator('Generator')
		(generated_img, VIS_DE, generate_IR) = G.transform(vis = SOURCE_VIS, ir = SOURCE_ir)
		print('generate:', generated_img.shape)
		g0 = tf.nn.avg_pool(generated_img, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
		generated_img_ds = tf.nn.avg_pool(g0, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

		D1 = Discriminator1('Discriminator1')
		grad_of_vis = grad(SOURCE_VIS)
		D1_real = D1.discrim(SOURCE_VIS, reuse = False)
		D1_fake = D1.discrim(generated_img, reuse = True)
		# source_shuffle = tf.random_shuffle(source)
		# D1_fake2 = D1.discrim(source_shuffle[:, :, :, 0], source[:, :, :, 1], reuse = True)

		D2 = Discriminator2('Discriminator2')
		D2_real = D2.discrim(SOURCE_ir, reuse = False)
		D2_fake = D2.discrim(generated_img_ds, reuse = True)

		#######  LOSS FUNCTION
		# Loss for Generator
		G_loss_GAN_D1 = -tf.reduce_mean(tf.log(D1_fake + eps))
		G_loss_GAN_D2 = -tf.reduce_mean(tf.log(D2_fake + eps))
		# G_loss_GAN_D1 = tf.reduce_mean(
		# 	tf.nn.sigmoid_cross_entropy_with_logits(logits = D1_fake,
		# 	                                        labels = tf.random_uniform(D1_real.shape, minval = 0.7,
		# 	                                                                   maxval = 1.2, dtype = tf.float32)))
		# G_loss_GAN_D2 = tf.reduce_mean(
		# 	tf.nn.sigmoid_cross_entropy_with_logits(logits = D2_fake,
		# 	                                        labels = tf.random_uniform(D2_real.shape, minval = 0.7,
		# 	                                                                   maxval = 1.2, dtype = tf.float32)))
		G_loss_GAN = G_loss_GAN_D1  + G_loss_GAN_D2
		# SSIM_VIS = SSIM_LOSS(tf.expand_dims(source[:, :, :, 0], -1), generated_img, size = 11, sigma = 1.5)
		# SSIM_IR = SSIM_LOSS(tf.expand_dims(source[:, :, :, 1], -1), generated_img, size = 11, sigma = 1.5)
		# G_loss_ssim = 1 - tf.reduce_mean(0.5 * SSIM_VIS + 0.5 * SSIM_IR)
		# generated_img_ds = tf.image.resize_images(generated_img,
		#                                           size = (BATCH_SIZE, patch_size / rc, patch_size / rc, 1), method = 2)

		LOSS_IR = Fro_LOSS(generated_img_ds - SOURCE_ir)
		LOSS_VIS = L1_LOSS(grad(generated_img) - grad_of_vis)
		G_loss_norm = LOSS_IR + 1.2 * LOSS_VIS  # - 0.5 * L1_LOSS(grad(generated_img))
		G_loss = G_loss_GAN + 0.5 * G_loss_norm

		# Loss for Discriminator1
		D1_loss_real = -tf.reduce_mean(tf.log(D1_real + eps))
		D1_loss_fake = -tf.reduce_mean(tf.log(1. - D1_fake + eps))
		# D1_loss_real = tf.reduce_mean(
		# 	tf.nn.sigmoid_cross_entropy_with_logits(logits = D1_real,
		# 	                                        labels = tf.random_uniform(D1_real.shape, minval = 0.7,
		# 	                                                                   maxval = 1.2, dtype = tf.float32)))
		# D1_loss_fake = tf.reduce_mean(
		# 	tf.nn.sigmoid_cross_entropy_with_logits(logits = D1_fake,
		# 	                                        labels = tf.random_uniform(D1_fake.shape, minval = 0.,
		# 	                                                                   maxval = 0.3, dtype = tf.float32)))
		D1_loss = D1_loss_fake + D1_loss_real

		# Loss for Discriminator2
		D2_loss_real = -tf.reduce_mean(tf.log(D2_real + eps))
		D2_loss_fake = -tf.reduce_mean(tf.log(1. - D2_fake + eps))
		# D2_loss_real = tf.reduce_mean(
		# 	tf.nn.sigmoid_cross_entropy_with_logits(logits = D2_real,
		# 	                                        labels = tf.random_uniform(D2_real.shape, minval = 0.7,
		# 	                                                                   maxval = 1.2, dtype = tf.float32)))
		# D2_loss_fake = tf.reduce_mean(
		# 	tf.nn.sigmoid_cross_entropy_with_logits(logits = D2_fake,
		# 	                                        labels = tf.random_uniform(D2_fake.shape, minval = 0.,
		# 	                                                                   maxval = 0.3, dtype = tf.float32)))
		D2_loss = D2_loss_fake + D2_loss_real

		current_iter = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE, global_step = current_iter,
		                                           decay_steps = int(n_batches), decay_rate = DECAY_RATE,
		                                           staircase = False)

		# theta_de = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'deconv_ir')
		theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
		theta_D1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator1')
		theta_D2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator2')

		G_GAN_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss_GAN, global_step = current_iter,
		                                                                 var_list = theta_G)
		G_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss, global_step = current_iter,
		                                                             var_list = theta_G)
		D1_solver = tf.train.GradientDescentOptimizer(learning_rate).minimize(D1_loss, global_step = current_iter,
		                                                                      var_list = theta_D1)
		D2_solver = tf.train.GradientDescentOptimizer(learning_rate).minimize(D2_loss, global_step = current_iter,
		                                                                      var_list = theta_D2)

		clip_G = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_G]
		clip_D1 = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_D1]
		clip_D2 = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_D2]

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(max_to_keep = 500)

		tf.summary.scalar('G_Loss_D1', G_loss_GAN_D1)
		tf.summary.scalar('G_Loss_D2', G_loss_GAN_D2)
		tf.summary.scalar('G_loss_norm', G_loss_norm)
		tf.summary.scalar('D1_real', tf.reduce_mean(D1_real))
		tf.summary.scalar('D1_fake', tf.reduce_mean(D1_fake))
		tf.summary.scalar('D2_real', tf.reduce_mean(D2_real))
		tf.summary.scalar('D2_fake', tf.reduce_mean(D2_fake))
		tf.summary.scalar('D1_loss', D1_loss)
		tf.summary.scalar('D2_loss', D2_loss)
		tf.summary.image('generated_img', generated_img)
		tf.summary.scalar('Learning rate', learning_rate)
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("logs/", sess.graph)

		# ** Start Training **
		step = 0
		count_loss = 0
		num_imgs = source_imgs.shape[0]

		for epoch in range(EPOCHS):
			np.random.shuffle(source_imgs)
			for batch in range(n_batches):
				ir_batch = np.zeros(shape = (BATCH_SIZE, int(patch_size / rc), int(patch_size / rc)),
				                    dtype = np.float32)
				step += 1
				current_iter = step
				VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
				ir_or_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
				for b in range(BATCH_SIZE):
					ir_batch[b, :, :] = scipy.ndimage.zoom(ir_or_batch[b, :, :], 0.25)
				VIS_batch = np.expand_dims(VIS_batch, -1)
				ir_batch = np.expand_dims(ir_batch, -1)
				FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_ir: ir_batch}

				it_g = 0
				it_d1 = 0
				it_d2 = 0
				# run the training step
				if batch % 2 == 0:
					sess.run([D1_solver, clip_D1], feed_dict = FEED_DICT)
					it_d1 += 1
					sess.run([D2_solver, clip_D2], feed_dict = FEED_DICT)
					it_d2 += 1
				else:
					sess.run([G_solver, clip_G], feed_dict = FEED_DICT)
					it_g += 1
				g_loss, d1_loss, d2_loss = sess.run([G_loss, D1_loss, D2_loss], feed_dict = FEED_DICT)

				if batch % 2 == 0:
					while d1_loss > 1.8 and it_d1 < 20:
						sess.run([D1_solver, clip_D1], feed_dict = FEED_DICT)
						d1_loss = sess.run(D1_loss, feed_dict = FEED_DICT)
						it_d1 += 1
					while d2_loss > 1.8 and it_d2 < 20:
						sess.run([D2_solver, clip_D2], feed_dict = FEED_DICT)
						d2_loss = sess.run(D2_loss, feed_dict = FEED_DICT)
						it_d2 += 1
				else:
					while (d1_loss < 1.2 or d2_loss < 1.2) and it_g < 20:
						sess.run([G_GAN_solver, clip_G], feed_dict = FEED_DICT)
						g_loss, d1_loss, d2_loss = sess.run([G_loss, D1_loss, D2_loss], feed_dict = FEED_DICT)
						it_g += 1
					while (g_loss > 200) and it_g < 20:
						sess.run([G_solver, clip_G], feed_dict = FEED_DICT)
						g_loss = sess.run(G_loss, feed_dict = FEED_DICT)
						it_g += 1
				print("%s %s D1:%s D2:%s, G:%s" % (epoch, batch, it_d1, it_d2, it_g))
				# if batch % 10 == 0:
				# 	elapsed_time = datetime.now() - start_time
				# 	lr = sess.run(learning_rate)
				# 	print("it_d1:%s, it_d2:%s, it_g:%s" % (it_d1, it_d2, it_g))
				# 	print('G_loss:%s, D1_loss:%s, D2_loss:%s, batch: %s/%s' % (
				# 		g_loss, d1_loss, d2_loss, batch, n_batches))
				# 	print("lr:%s, elapsed_time:%s" % (lr, elapsed_time))

				# if batch % 2 == 0:
				# 	while (d1_loss > 1.9 or (np.abs(d1_loss - d1_past) < 0.0002)) and it_d1 < 20:
				# 		sess.run([D1_solver, clip_D1], feed_dict = FEED_DICT)
				# 		d1_loss = sess.run(D1_loss, feed_dict = FEED_DICT)
				# 		it_d1 += 1
				# 	while (d2_loss > 1.9 or (np.abs(d2_loss - d2_past) < 0.0002)) and it_d2 < 20:
				# 		sess.run([D2_solver, clip_D2], feed_dict = FEED_DICT)
				# 		d2_loss = sess.run(D2_loss, feed_dict = FEED_DICT)
				# 		it_d2 += 1
				# 	while (d1_loss < 1 or d2_loss < 1) and it_g < 10:
				# 		sess.run([G_GAN_solver, clip_G], feed_dict = FEED_DICT)
				# 		g_loss, d1_loss, d2_loss = sess.run([G_loss, D1_loss, D2_loss], feed_dict = FEED_DICT)
				# 		it_g += 1
				# 	while g_loss > 100 and it_g < 20:
				# 		sess.run([G_solver, clip_G], feed_dict = FEED_DICT)
				# 		g_loss = sess.run(G_loss, feed_dict = FEED_DICT)
				# 		it_g += 1
				# if batch % 10 == 0:
				# 	elapsed_time = datetime.now() - start_time
				# 	lr = sess.run(learning_rate)
				# 	print("it_d1:%s, it_d2:%s, it_g:%s" % (it_d1, it_d2, it_g))
				# 	print('G_loss:%s, D1_loss:%s, D2_loss:%s, batch: %s/%s' % (
				# 		g_loss, d1_loss, d2_loss, batch, n_batches))
				# 	print("lr:%s, elapsed_time:%s" % (lr, elapsed_time))

				result = sess.run(merged, feed_dict = FEED_DICT)
				writer.add_summary(result, step)
				if step % logging_period == 0:
					saver.save(sess, save_path + str(step) + '/' + str(step) + '.ckpt')
				# gofvis = sess.run(grad_of_vis, feed_dict = {source: source_batch})
				# IF, gofIF = sess.run([generated_img, grad(generated_img)], feed_dict = {source: source_batch})
				# fig = plt.figure()
				# orig = fig.add_subplot(221)
				# gradslove = fig.add_subplot(222)
				# orig.imshow(source_batch[1, :, :, 0], cmap = 'gray')
				# gradslove.imshow(gofvis[1, :, :, 0], cmap = 'gray')
				# oriF = fig.add_subplot(223)
				# gradF = fig.add_subplot(224)
				# oriF.imshow(sess.run(generated_img, feed_dict = {source: source_batch})[1, :, :, 0], cmap = 'gray')
				# print(sess.run(grad(generated_img)[1, :, :, 0], feed_dict = {source: source_batch}))
				# gradF.imshow(sess.run(grad(generated_img)[1, :, :, 0], feed_dict = {source: source_batch}),
				#              cmap = 'gray')
				# plt.show()



				is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)
				if is_last_step or step % logging_period == 0:
					elapsed_time = datetime.now() - start_time
					g_loss_GAN, g_loss_norm = sess.run([G_loss_GAN, G_loss_norm], feed_dict = FEED_DICT)
					g_loss, d1_loss = sess.run([G_loss, D1_loss], feed_dict = FEED_DICT)
					d1_fake, d1_real = sess.run([tf.reduce_mean(D1_fake), tf.reduce_mean(D1_real)],
					                            feed_dict = FEED_DICT)
					d2_fake, d2_real = sess.run([tf.reduce_mean(D2_fake), tf.reduce_mean(D2_real)],
					                            feed_dict = FEED_DICT)
					lr = sess.run(learning_rate)
					loss_vis = sess.run(LOSS_VIS, feed_dict = FEED_DICT)
					loss_ir = sess.run(LOSS_IR, feed_dict = FEED_DICT)
					g_img = sess.run(generated_img, feed_dict = FEED_DICT)

					print('epoch:%d/%d, step:%d, lr:%s, elapsed_time:%s' % (
						epoch + 1, EPOCHS, step, lr, elapsed_time))
					print('D1_fake:%s, D1_real:%s' % (d1_fake, d1_real))
					print('D2_fake:%s, D2_real:%s' % (d2_fake, d2_real))
					# print('G_loss:%s, D1_loss:%s, D2_loss:%s' % (g_loss, d1_loss, d2_loss))
					# print('G_loss_gan: %s, G_loss_norm: %s' % (g_loss_GAN, g_loss_norm))
					# print('loss_vis: %s, loss_ir: %s' % (loss_vis, loss_ir))
					maxv = np.max(g_img)
					minv = np.min(g_img)
					print('max_value:%s, min_value:%s\n' % (maxv, minv))

				# if epoch % 1 == 0:
				# 	saver.save(sess, MODEL_SAVE_PATH)
				# 	for i in range(9):
				# 		index = i + 1
				# 		ir_path = path + 'IR_ds_us' + str(index) + '.bmp'
				# 		vis_path = path + 'VIS' + str(index) + '.bmp'
				# 		generate(ir_path, vis_path, MODEL_SAVE_PATH, index, output_path = 'Fused images/' + str(epoch) + '/')

		writer.close()
		saver.save(sess, save_path + str(epoch) + '/' + str(epoch) + '.ckpt')


# 		# current_iter = tf.Variable(0)
# 		# learning_rate = tf.train.exponential_decay(0.03, current_iter, decay_steps = n_batches, decay_rate = 0.03)
# 		# train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step = current_iter)
#

# 	# ** Done Training & Save the model **
# 	saver.save(sess, save_path)

# loss_data = Loss_all[:count_loss]
# scio.savemat('./models/loss/only_ssim.mat', {'loss': loss_data})


def grad(img):
	# [n, r, c, channel] = [int(img.shape[0]), int(img.shape[1]), int(img.shape[2]), int(img.shape[3])]
	# img_h = tf.slice(img, [0, 0, 1, 0], [n, r, c - 1, channel])
	# img_v = tf.slice(img, [0, 1, 0, 0], [n, r - 1, c, channel])
	# zeros_h = tf.zeros([n, r, 1, channel])
	# zeros_v = tf.zeros([n, 1, c, channel])
	# img_h = tf.concat([img_h, zeros_h], axis = 2)
	# img_v = tf.concat([img_v, zeros_v], axis = 1)
	# g_h = img - img_h
	# g_v = img - img_v
	# g = tf.sqrt(tf.square(g_h) + tf.square(g_v))

	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, tf.cast(kernel,tf.float32), strides = [1, 1, 1, 1], padding = 'SAME')
	# g = tf.nn.tanh(g) / 2 + 0.5
	return g
