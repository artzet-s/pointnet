# ==============================================================================
import numpy as np
import tensorflow as tf
import argparse
import os
import sys
# ==============================================================================
# local import

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

import numpy
import glob
import model
import indoor3d_util
# ==============================================================================

BATCH_SIZE = 1
NUM_POINT = 6144
MODEL_PATH = "/ext_data/artzet_s/log_apple_5_normalized/model_100.ckpt"
GPU_INDEX = 0
DUMP_DIR = "/ext_data/artzet_s/log_apple_5_normalized/dump"
NUM_CLASSES = 2

if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')

# ==============================================================================


def log_string(out_str):
	LOG_FOUT.write(out_str + '\n')
	LOG_FOUT.flush()
	print(out_str)


def evaluate(input_folder, output_folder):

	with tf.device('/gpu:' + str(GPU_INDEX)):
		# ======================================================================
		# Set the placeholder tensor
		pointclouds_pl = tf.placeholder(
			tf.float32,
			shape=(BATCH_SIZE, NUM_POINT, 9))

		labels_pl = tf.placeholder(
			tf.int32,
			shape=(BATCH_SIZE, NUM_POINT))

		is_training_pl = tf.placeholder(
			tf.bool,
			shape=())

		# ======================================================================
		# Set model
		pred = model.get_model(pointclouds_pl, is_training_pl)

		pred_softmax = tf.nn.softmax(pred)

		# ======================================================================
		# Set loss
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=pred,
			labels=labels_pl)
		loss = tf.reduce_mean(loss)

		# ======================================================================
		# Add ops to save and restore all the variables.
		saver = tf.train.Saver()

	# ==========================================================================
	# Create a session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = True
	sess = tf.Session(config=config)

	# ==========================================================================
	# Restore variables from disk.
	saver.restore(sess, MODEL_PATH)
	log_string("Model restored.")

	# ==========================================================================
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

		eval_one_epoch(sess,
		               ops,
		               filename,
		               output_filename)


def eval_one_epoch(sess, ops, input_filename, output_filename):
	# ==========================================================================
	is_training = False
	total_correct = 0
	total_seen = 0
	loss_sum = 0
	total_seen_class = [0 for _ in range(NUM_CLASSES)]
	total_correct_class = [0 for _ in range(NUM_CLASSES)]

	# ==========================================================================

	data_label = np.loadtxt(input_filename)
	min_max = np.loadtxt("mean_data.txt")

	current_data, current_label = indoor3d_util.my_room2blocks_plus_normalized(
		data_label,
	    NUM_POINT,
		min_max,
		block_size=0.5,
		stride=0.5,
		test_mode=True)
	print(current_label.shape)
	current_label[:, :] = 0
	# ==========================================================================

	current_data = current_data[:, 0:NUM_POINT, :]

	print("Current data shape : {} ".format(current_data.shape))
	print("Current label shape : {} ".format(current_label.shape))

	# ==========================================================================
	# Get room dimension..

	print("data label shape : {} ".format(data_label.shape))

	max_room_x = max(data_label[:, 0])
	max_room_y = max(data_label[:, 1])
	max_room_z = max(data_label[:, 2])

	list_point = list()

	file_size = current_data.shape[0]
	num_batches = file_size // BATCH_SIZE
	print(file_size, BATCH_SIZE, num_batches)

	for batch_idx in range(num_batches):

		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx + 1) * BATCH_SIZE
		cur_batch_size = end_idx - start_idx

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
		for b in range(BATCH_SIZE):
			pts = current_data[start_idx + b, :, :]
			# pts[:, 6] *= max_room_x
			# pts[:, 7] *= max_room_y
			# pts[:, 8] *= max_room_z

			pred = pred_label[b, :]

			for i in range(NUM_POINT):
				list_point.append((pts[i, 0], pts[i, 1], pts[i, 2], pred[i]))

		correct = np.sum(pred_label == current_label[start_idx:end_idx, :])
		total_correct += correct
		total_seen += (cur_batch_size * NUM_POINT)
		loss_sum += (loss_val * BATCH_SIZE)

		for i in range(start_idx, end_idx):
			for j in range(NUM_POINT):
				l = current_label[i, j]
				total_seen_class[l] += 1

				total_correct_class[l] += (pred_label[i - start_idx, j] == l)

	a = np.array(list_point)
	a = np.unique(a, axis=0)
	np.savetxt(output_filename, a, delimiter=' ')

	log_string(
		'eval mean loss: %f' % (loss_sum / float(total_seen / NUM_POINT)))
	log_string('eval accuracy: %f' % (total_correct / float(total_seen)))

	return total_correct, total_seen


def train_test_evaluate():

	input_folder = "/ext_data/artzet_s/afef_apple_tree_filtred"
	output_folder = "/ext_data/artzet_s/eval_3"

	if not os.path.exists(output_folder): os.mkdir(output_folder)

	LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')

	with tf.Graph().as_default():
		evaluate(input_folder, output_folder)

	LOG_FOUT.close()


if __name__ == '__main__':
	main()
