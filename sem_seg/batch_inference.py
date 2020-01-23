#
#
#       EVALUATE NN MODEL
#
#
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

import model
import indoor3d_util
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--dump_dir', required=True, help='dump folder path')
parser.add_argument('--room_data_filelist', required=True, help='TXT filename, filelist, each line is a test room data label file.')
parser.add_argument('--no_clutter', action='store_true', help='If true, donot count the clutter class')
FLAGS = parser.parse_args()
# ==============================================================================

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
INPUT_FILENAME_LIST = [os.path.join(ROOT_DIR, line.rstrip()) for line in open(
    FLAGS.room_data_filelist)]

NUM_CLASSES = 13
# ==============================================================================


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate():

    is_training = False
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

    total_correct, total_seen = 0, 0
    for input_filename in INPUT_FILENAME_LIST:

        output_filename = os.path.basename(input_filename)[:-4] + '_pred.txt'
        output_filename = os.path.join(DUMP_DIR, output_filename)

        a, b = eval_one_epoch(sess,
                              ops,
                              input_filename,
                              output_filename)

        total_correct += a
        total_seen += b

    log_string('All eval accuracy: %f' % (total_correct / float(total_seen)))


def eval_one_epoch(sess, ops, input_filename, output_filename):

    # ==========================================================================
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    # ==========================================================================
    #
    current_data, current_label = indoor3d_util.room2blocks_wrapper_normalized(
        input_filename,
        NUM_POINT,
        test_mode=True)

    current_data = current_data[:, 0:NUM_POINT, :]
    current_label = np.squeeze(current_label)

    print("Current data shape : {} ".format(current_data.shape))
    print("Current label shape : {} ".format(current_label.shape))

    # ==========================================================================
    fout_data_label = open(output_filename, 'w')

    # Get room dimension..
    data_label = np.load(input_filename)
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
        if FLAGS.no_clutter:
            pred_label = np.argmax(pred_val[:, :, 0:12], 2) # BxN
        else:
            pred_label = np.argmax(pred_val, 2) # BxN

        # ======================================================================
        # Save prediction labels
        for b in range(BATCH_SIZE):
            pts = current_data[start_idx + b, :, :]
            pts[:, 6] *= max_room_x
            pts[:, 7] *= max_room_y
            pts[:, 8] *= max_room_z
            pts[:, 3:6] *= 255.0
            pred = pred_label[b, :]

            for i in range(NUM_POINT):
                # line = '%f %f %f %d %d %d %f %d\n' % (
                #     pts[i, 6], pts[i, 7], pts[i, 8],
                #     pts[i, 3], pts[i, 4], pts[i, 5],
                #     pred_val[b, i, pred[i]],
                #     pred[i])

                list_point.append((pts[i, 0], pts[i, 1], pts[i, 2], pred[i]))

                # fout_data_label.write(line)

        correct = np.sum(pred_label == current_label[start_idx:end_idx, :])
        total_correct += correct
        total_seen += (cur_batch_size * NUM_POINT)
        loss_sum += (loss_val * BATCH_SIZE)

        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):

                l = current_label[i, j]

                total_seen_class[l] += 1

                total_correct_class[l] += (pred_label[i-start_idx, j] == l)

    a = np.array(list_point)
    print(a.shape, a)
    a = np.unique(a, axis=0)
    print(a.shape, a)
    np.savetxt(output_filename, a, delimiter=' ')

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))

    # fout_data_label.close()

    return total_correct, total_seen


def main():

    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()


if __name__=='__main__':
    main()
