# ==============================================================================
import argparse
import numpy as np
import tensorflow as tf
import socket
# ==============================================================================
import os
import sys
import glob
import numpy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import provider
import model
import indoor3d_util
import organise_apple_tree

# ==============================================================================

BATCH_SIZE = 24
NUM_POINT = 6144  # (base) 4096
MAX_EPOCH = 50  # (base) 50
BASE_LEARNING_RATE = 0.001  # (base) 0.001
GPU_INDEX = 0
MOMENTUM = 0.9
OPTIMIZER = 'adam'
DECAY_STEP = 300000
DECAY_RATE = 0.5

# ==============================================================================

NUM_CLASSES = 2
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
# BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# ==============================================================================


def get_data(folder):
	data_batch_list = []
	label_batch_list = []

	for filename in glob.glob("{}/*.npy".format(folder)):
		print(filename)
		data = numpy.load(filename)
		data_batch_list.append(data[:, :, :-1])
		label_batch_list.append(data[:, :, -1])

	data_batches = np.concatenate(data_batch_list, 0)
	label_batches = np.concatenate(label_batch_list, 0).astype(numpy.uint8)

	print(data_batches.shape, label_batches.shape)
	return data_batches, label_batches


class Log(object):

	directory = None
	log_fout = None

	def __init__(self, directory, log_train="log_train.txt"):

		self.directory = directory

		if not os.path.exists(self.directory):
			os.mkdir(self.directory)

		os.system('cp model.py {}'.format(self.directory))
		os.system('cp train_test_evaluate.py {}'.format(self.directory))

		self.log_fout = open(os.path.join(self.directory, log_train), 'w')

	def log_string(self, out_str):
		self.log_fout.write(out_str + '\n')
		self.log_fout.flush()
		print(out_str)

	def __del__(self):
		self.log_fout.close()

# ==============================================================================


def train(train_data, train_label, test_data, test_label, log, K=9):

	print("Num GPUs Available: ",
	      len(tf.config.experimental.list_physical_devices('GPU')))

	tf.debugging.set_log_device_placement(True)
	strategy = tf.distribute.MirroredStrategy()

	# train_data = strategy.experimental_distribute_dataset(train_data)
	# train_label = strategy.experimental_distribute_dataset(train_label)
	# test_data = strategy.experimental_distribute_dataset(test_data)
	# test_label = strategy.experimental_distribute_dataset(test_label)

	with strategy.scope():
		with tf.Graph().as_default():

			# ==================================================================
			# Set the placeholder tensor
			pointclouds_pl = tf.placeholder(
				tf.float32,
				shape=(BATCH_SIZE, NUM_POINT, K))

			labels_pl = tf.placeholder(
				tf.int32,
				shape=(BATCH_SIZE, NUM_POINT))

			is_training_pl = tf.placeholder(
				tf.bool,
				shape=())

			# ==================================================================
			# Note the global_step=batch parameter to minimize.
			# That tells the optimizer to helpfully increment the 'batch'
			# parameter for you every time it trains.
			batch = tf.Variable(0)

			# ==================================================================
			# SET DECAY
			bn_momentum = tf.train.exponential_decay(
				BN_INIT_DECAY,
				batch * BATCH_SIZE,
				BN_DECAY_DECAY_STEP,
				BN_DECAY_DECAY_RATE,
				staircase=True)

			bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
			tf.summary.scalar('bn_decay', bn_decay)

			# ==================================================================
			# Set model

			pred = model.get_model(pointclouds_pl,
			                       is_training_pl,
			                       bn_decay=bn_decay,
			                       K=K)

			# ==================================================================
			# Set loss
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=pred,
				labels=labels_pl)
			loss = tf.reduce_mean(loss)
			tf.summary.scalar('loss', loss)

			# ==================================================================
			# Set Accurancy
			correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))

			accuracy = tf.reduce_sum(
				tf.cast(correct, tf.float32)) / float(BATCH_SIZE * NUM_POINT)

			tf.summary.scalar('accuracy', accuracy)

			# ==================================================================
			# Set learning_rate
			learning_rate = tf.train.exponential_decay(
				BASE_LEARNING_RATE,
				batch * BATCH_SIZE,  # Current index into the dataset.
				DECAY_STEP,
				DECAY_RATE,
				staircase=True)

			# CLIP THE LEARNING RATE!!
			learning_rate = tf.maximum(learning_rate, 0.00001)
			tf.summary.scalar('learning_rate', learning_rate)

			# ==================================================================
			# Merge all the Summary value
			merged = tf.summary.merge_all()

			# ==================================================================
			# Set training optimizer
			if OPTIMIZER == 'momentum':
				optimizer = tf.train.MomentumOptimizer(
					learning_rate, momentum=MOMENTUM)
			elif OPTIMIZER == 'adam':
				optimizer = tf.train.AdamOptimizer(learning_rate)
			train_op = optimizer.minimize(loss, global_step=batch)

			# ==================================================================
			# Add ops to save and restore all the variables.
			saver = tf.train.Saver()

			# ==================================================================
			# Create a session
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			config.allow_soft_placement = True
			config.log_device_placement = True
			sess = tf.Session(config=config)

			# ==================================================================
			# Set file writer

			train_writer = tf.summary.FileWriter(
				os.path.join(log.directory, 'train'), sess.graph)

			test_writer = tf.summary.FileWriter(
				os.path.join(log.directory, 'test'))

			# ==================================================================
			# Init variables
			init = tf.global_variables_initializer()
			sess.run(init, {is_training_pl: True})

			ops = {'pointclouds_pl': pointclouds_pl,
			       'labels_pl': labels_pl,
			       'is_training_pl': is_training_pl,
			       'pred': pred,
			       'loss': loss,
			       'train_op': train_op,
			       'merged': merged,
			       'step': batch}

			for epoch in range(MAX_EPOCH + 1):
				log.log_string('**** EPOCH %03d ****' % epoch)
				sys.stdout.flush()

				train_one_epoch(train_data,
				                train_label,
				                sess,
				                ops,
				                train_writer,
				                log)

				test_one_epoch(test_data,
				               test_label,
				               sess,
				               ops,
				               test_writer,
				               log)

				# Save the variables to disk.
				if epoch % 10 == 0:
					save_path = saver.save(
						sess,
						os.path.join(log.directory, "model_{}.ckpt".format(epoch)))

					log.log_string("Model saved in file: %s" % save_path)


def train_one_epoch(train_data, train_label, sess, ops, train_writer, log):
	""" ops: dict mapping from string to tf ops """
	is_training = True

	log.log_string('---- TRAIN')
	current_data, current_label, _ = provider.shuffle_data(
		train_data[:, 0:NUM_POINT, :],
		train_label)

	file_size = current_data.shape[0]
	num_batches = file_size // BATCH_SIZE

	total_correct = 0
	total_seen = 0
	loss_sum = 0

	for batch_idx in range(num_batches):
		if batch_idx % 100 == 0:
			print('Current batch/total batch num: %d/%d' % (
				batch_idx, num_batches))
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx + 1) * BATCH_SIZE

		feed_dict = {
			ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
			ops['labels_pl']: current_label[start_idx:end_idx],
			ops['is_training_pl']: is_training}

		summary, step, _, loss_val, pred_val = sess.run(
			[ops['merged'],
			 ops['step'],
			 ops['train_op'],
			 ops['loss'],
			 ops['pred']], feed_dict=feed_dict)

		train_writer.add_summary(summary, step)
		pred_val = np.argmax(pred_val, 2)
		correct = np.sum(pred_val == current_label[start_idx:end_idx])
		total_correct += correct
		total_seen += (BATCH_SIZE * NUM_POINT)
		loss_sum += loss_val

	log.log_string('mean loss: %f' % (loss_sum / float(num_batches)))
	log.log_string('accuracy: %f' % (total_correct / float(total_seen)))


def test_one_epoch(test_data, test_label, sess, ops, test_writer, log):
	""" ops: dict mapping from string to tf ops """
	is_training = False

	total_correct = 0
	total_seen = 0
	loss_sum = 0

	total_seen_class = [0 for _ in range(NUM_CLASSES)]
	total_correct_class = [0 for _ in range(NUM_CLASSES)]

	log.log_string('---- TEST')
	current_data = test_data[:, 0:NUM_POINT, :]
	current_label = np.squeeze(test_label)

	file_size = current_data.shape[0]
	num_batches = file_size // BATCH_SIZE

	for batch_idx in range(num_batches):
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx + 1) * BATCH_SIZE

		feed_dict = {
			ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
			ops['labels_pl']: current_label[start_idx:end_idx],
			ops['is_training_pl']: is_training}

		summary, step, loss_val, pred_val = sess.run(
			[ops['merged'],
			 ops['step'],
			 ops['loss'],
			 ops['pred']],
			feed_dict=feed_dict)

		test_writer.add_summary(summary, step)
		pred_val = np.argmax(pred_val, 2)
		correct = np.sum(pred_val == current_label[start_idx:end_idx])

		total_correct += correct
		total_seen += (BATCH_SIZE * NUM_POINT)
		loss_sum += (loss_val * BATCH_SIZE)

		for i in range(start_idx, end_idx):
			for j in range(NUM_POINT):
				l = current_label[i, j]
				total_seen_class[l] += 1
				total_correct_class[l] += (pred_val[i - start_idx, j] == l)

	log.log_string(
		'eval mean loss: %f' % (loss_sum / float(total_seen / NUM_POINT)))
	log.log_string('eval accuracy: %f' % (total_correct / float(total_seen)))

	for l in range(NUM_CLASSES):
		log.log_string('eval {} class accuracy: {}'.format(
			l, total_correct_class[l] / float(total_seen_class[l])))

	log.log_string('eval avg class acc: %f' % (
		np.mean(
			np.array(total_correct_class) /
			np.array(total_seen_class, dtype=np.float))))

# ==============================================================================


def evaluate(input_folder, output_folder, model_path, log, K=9):

	with tf.Graph().as_default():
		with tf.device('/gpu:' + str(GPU_INDEX)):
			# ==================================================================
			# Set the placeholder tensor
			pointclouds_pl = tf.placeholder(
				tf.float32,
				shape=(1, NUM_POINT, K))

			labels_pl = tf.placeholder(
				tf.int32,
				shape=(1, NUM_POINT))

			is_training_pl = tf.placeholder(
				tf.bool,
				shape=())

			# ==================================================================
			# Set model
			pred = model.get_model(pointclouds_pl, is_training_pl, K=K)

			pred_softmax = tf.nn.softmax(pred)

			# ==================================================================
			# Set loss
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=pred,
				labels=labels_pl)
			loss = tf.reduce_mean(loss)

			# ==================================================================
			# Add ops to save and restore all the variables.
			saver = tf.train.Saver()

		# ======================================================================
		# Create a session
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		config.log_device_placement = True
		sess = tf.Session(config=config)

		# ======================================================================
		# Restore variables from disk.
		saver.restore(sess, model_path)
		log.log_string("Model restored.")

		# ======================================================================
		ops = {'pointclouds_pl': pointclouds_pl,
		       'labels_pl': labels_pl,
		       'is_training_pl': is_training_pl,
		       'pred': pred,
		       'pred_softmax': pred_softmax,
		       'loss': loss}

		for filename in glob.glob("{}/*.txt".format(input_folder)):
			print(filename)

			output_filename = os.path.basename(filename)[:-4] + '_pred.txt'
			output_filename = os.path.join(output_folder, output_filename)

			eval_one_epoch(filename,
			               output_filename,
			               sess,
			               ops)

			print(output_filename)


def eval_one_epoch(input_filename, output_filename, sess, ops):
	# ==========================================================================
	is_training = False

	data_label = np.loadtxt(input_filename)
	min_max = np.loadtxt("mean_data.txt")
	current_data, current_label = indoor3d_util.my_room2blocks_plus_normalized(
		data_label,
		NUM_POINT,
		min_max,
		block_size=0.5,
		stride=0.5,
		test_mode=True)

	# current_data, current_label = organise_apple_tree.block_xyz(
	# 	data_label,
	# 	NUM_POINT,
	# 	block_size=0.5,
	# 	stride=0.5,
	# 	test_mode=True)

	current_label[:, :] = 0
	# ==========================================================================

	list_point = list()
	for batch_idx in range(current_data.shape[0]):

		start_idx = batch_idx
		end_idx = (batch_idx + 1)

		feed_dict = {
			ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
			ops['labels_pl']: current_label[start_idx:end_idx],
			ops['is_training_pl']: is_training}

		loss_val, pred_val = sess.run(
			[ops['loss'],
			 ops['pred_softmax']], feed_dict=feed_dict)

		# ======================================================================
		# Take noise or not ?
		pred_label = np.argmax(pred_val, 2)  # BxN
		# ======================================================================
		# Save prediction labels
		pts = current_data[start_idx, :, :]
		pred = pred_label[0, :]

		for i in range(NUM_POINT):
			list_point.append((pts[i, 0], pts[i, 1], pts[i, 2], pred[i]))

	a = np.unique(np.array(list_point), axis=0)
	np.savetxt(output_filename, a, delimiter=' ')


def run_train_test_evaluate():

	train_data, train_label = get_data(
		'/ext_data/artzet_s/train_floor_data')

	test_data, test_label = get_data(
		'/ext_data/artzet_s/test_floor_data')

	log = Log("/ext_data/artzet_s/log_model_floor")
	# train(train_data, train_label, test_data, test_label, log, K=9)

	input_folder = "/ext_data/artzet_s/afef_apple_tree_filtred_floor"
	output_folder = "/ext_data/artzet_s/eval_floor"
	model_path = "/ext_data/artzet_s/log_apple_5_normalized/model_50.ckpt"
	# model_path = "/ext_data/artzet_s/log_model/model_50.ckpt"

	if not os.path.exists(output_folder): os.mkdir(output_folder)
	evaluate(input_folder, output_folder, model_path, log, K=9)


def run_train_test_evaluate_synth():

	train_data, train_label = get_data(
		'/ext_data/artzet_s/train_synthetic_data')

	test_data, test_label = get_data(
		'/ext_data/artzet_s/test_synthetic_data')

	log = Log("/ext_data/artzet_s/log_synthetic_model")

	train(train_data, train_label, test_data, test_label, log, K=6)

	input_folder = "/ext_data/artzet_s/synthetic_lidar_simulation"
	output_folder = "/ext_data/artzet_s/eval_synthetic_data"
	model_path = "/ext_data/artzet_s/log_synthetic_model/model_50.ckpt"

	if not os.path.exists(output_folder): os.mkdir(output_folder)
	evaluate(input_folder, output_folder, model_path, log, K=6)


if __name__ == "__main__":
	run_train_test_evaluate()
	# run_train_test_evaluate_synth()
