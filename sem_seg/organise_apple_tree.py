import os
import numpy
import sys
import glob
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import data_prep_util
import indoor3d_util
import shutil
import random


def initialize_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)


def _get_indice_3d_windows(xyz, x0, xn, y0, yn, z0, zn):
    indx = numpy.bitwise_and(x0 <= xyz[:, 0], xyz[:, 0] < xn)
    indy = numpy.bitwise_and(y0 <= xyz[:, 1], xyz[:, 1] < yn)
    indz = numpy.bitwise_and(z0 <= xyz[:, 2], xyz[:, 2] < zn)

    return numpy.bitwise_and(numpy.bitwise_and(indx, indy), indz)


def split_3d_point_cloud_to_several_windows(point_cloud, window_size):
    xyz = point_cloud[:, :3]
    ws = numpy.array(window_size)
    xyz_max = numpy.max(xyz, axis=0)
    xyz_min = numpy.min(xyz, axis=0)
    pc_nb = numpy.ceil((xyz_max - xyz_min) / ws).astype(int)

    indices = list()
    for i in range(0, pc_nb[0]):
        for j in range(0, pc_nb[1]):
            for k in range(0, pc_nb[2]):
                x0, y0, z0 = xyz_min + ws * numpy.array([i, j, k])
                xn, yn, zn = xyz_min + ws * numpy.array([i + 1, j + 1, k + 1])

                ind = _get_indice_3d_windows(xyz, x0, xn, y0, yn, z0, zn)
                indices.append(ind)

    return indices


def compute_min_max():
    input_folder = "/home/artzet_s/code/dataset/afef_apple_tree_filtred"

    vmax, vmin = list(), list()
    for i, filename in enumerate(
            glob.glob("{}/pc_2018*.txt".format(input_folder))):
        data_label = numpy.loadtxt(filename)
        vmax.append(numpy.amax(data_label, axis=0))
        vmin.append(numpy.amin(data_label, axis=0))
    vmax = numpy.amax(numpy.array(vmax), axis=0)
    vmin = numpy.amax(numpy.array(vmin), axis=0)

    arr = numpy.stack([vmin, vmax], axis=0)
    numpy.savetxt("mean_data.txt", arr)


def build_blocks(data,
                 label,
                 num_point,
                 test_mode=False,
                 K=6):

    indices = split_3d_point_cloud_to_several_windows(
        data, window_size=(0.5, 0.5, 0.5))

    # Collect blocks
    block_data_list = []
    block_label_list = []

    for cond in indices:
        if numpy.count_nonzero(cond) < 1000:
            continue

        block_data = data[cond]
        block_label = label[cond]

        block_data_sampled, block_label_sampled = indoor3d_util.room2samples(
            block_data, block_label, num_point, K=K)

        if test_mode:
            for i in range(block_data_sampled.shape[0]):
                block_data_list.append(
                    numpy.expand_dims(block_data_sampled[i, :, ], 0))
                block_label_list.append(
                    numpy.expand_dims(block_label_sampled[i, :, 0], 0))
        else:
            # randomly subsample data
            block_data_sampled, block_label_sampled = indoor3d_util.sample_data_label(
                block_data, block_label, num_point)

            block_data_list.append(numpy.expand_dims(block_data_sampled, 0))
            block_label_list.append(numpy.expand_dims(block_label_sampled, 0))

    return numpy.concatenate(block_data_list, 0), numpy.concatenate(block_label_list,
                                                              0)


def block_xyzrad(data_label,
                 num_point,
                 min_max,
                 test_mode=False):

    data = data_label[:, :-1]
    label = data_label[:, -1].astype(numpy.uint8)

    # CENTRALIZE HERE
    data[:, :3] = data[:, :3] - numpy.amin(data, 0)[0:3]
    data[:, 3:6] = (data[:, 3:6] - min_max[0, 3:6]) / min_max[1, 3:6]

    data_batch, label_batch = build_blocks(data,
                                          label,
                                          num_point,
                                          test_mode,
                                           K=6)

    return data_batch, label_batch


def block_xyz(data_label,
              num_point,
              block_size=1.0,
              stride=1.0,
              random_sample=False,
              sample_num=None,
              sample_aug=1,
              test_mode=False):

    data = data_label[:, 0:3]
    label = data_label[:, -1].astype(numpy.uint8)

    # CENTRALIZE HERE
    data[:, :3] = data[:, :3] - numpy.amin(data, 0)[0:3]

    max_room_x = max(data[:, 0])
    max_room_y = max(data[:, 1])
    max_room_z = max(data[:, 2])

    data_batch, label_batch = indoor3d_util.room2blocks(data,
                                                        label,
                                                        num_point,
                                                        block_size,
                                                        stride,
                                                        random_sample,
                                                        sample_num,
                                                        sample_aug,
                                                        test_mode,
                                                        K=3)

    new_data_batch = numpy.zeros((data_batch.shape[0], num_point, 6))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 3] = data_batch[b, :, 0] / max_room_x
        new_data_batch[b, :, 4] = data_batch[b, :, 1] / max_room_y
        new_data_batch[b, :, 5] = data_batch[b, :, 2] / max_room_z

    new_data_batch[:, :, 0:3] = data_batch

    return new_data_batch, label_batch


def organize_data():

    input_folder = "/home/artzet_s/code/dataset/afef_apple_tree_filtred_labeled/aug"
    train_folder = "/home/artzet_s/code/dataset/train_block_data"
    test_folder = "/home/artzet_s/code/dataset/test_block_data"

    initialize_folder(train_folder)
    initialize_folder(test_folder)

    filenames = glob.glob("{}/*.txt".format(input_folder))
    min_max = numpy.loadtxt("mean_data.txt")

    for i, filename in enumerate(filenames):
        print("{}/{}".format(i, len(filenames)))

        data_label = numpy.loadtxt(filename)
        data, label = block_xyzrad(
            data_label,
            6144,
            min_max)

        label = numpy.array([label]).reshape((label.shape[0], label.shape[1], 1))
        data_label = numpy.concatenate([data, label], axis=2)
        basename = os.path.basename(filename)[:-4]

        if i < len(filenames) * 0.80:
            numpy.save("{}/{}.npy".format(train_folder, basename), data_label)
        else:
            numpy.save("{}/{}.npy".format(test_folder, basename), data_label)


def organize_floor_data():

    input_folder = "/home/artzet_s/code/dataset/afef_apple_tree_filtred_labeled/aug"
    train_folder = "/home/artzet_s/code/dataset/train_floor_data"
    test_folder = "/home/artzet_s/code/dataset/test_floor_data"

    initialize_folder(train_folder)
    initialize_folder(test_folder)

    filenames = glob.glob("{}/*.txt".format(input_folder))
    min_max = numpy.loadtxt("mean_data.txt")

    for i, filename in enumerate(filenames):
        print("{}/{}".format(i, len(filenames)))

        data_label = numpy.loadtxt(filename)
        data_label = data_label[data_label[:, 2] <= -1.40]
        data, label = indoor3d_util.my_room2blocks_plus_normalized(
            data_label,
            6144,
            min_max,
            block_size=0.5,
            stride=0.5,
            random_sample=False,
            sample_num=None)

        label = numpy.array([label]).reshape(
            (label.shape[0], label.shape[1], 1))
        data_label = numpy.concatenate([data, label], axis=2)
        basename = os.path.basename(filename)[:-4]

        if i < len(filenames) * 0.80:
            numpy.save("{}/{}.npy".format(train_folder, basename), data_label)
        else:
            numpy.save("{}/{}.npy".format(test_folder, basename), data_label)


def organize_synthetic_data():

    input_folder = "/home/artzet_s/code/dataset/synthetic_lidar_simulation"
    train_folder = "/home/artzet_s/code/dataset/train_synthetic_data"
    test_folder = "/home/artzet_s/code/dataset/test_synthetic_data"

    initialize_folder(train_folder)
    initialize_folder(test_folder)

    filenames = glob.glob("{}/*.txt".format(input_folder))
    random.shuffle(filenames)

    min_max = numpy.loadtxt("mean_data.txt")

    for i, filename in enumerate(filenames):
        print("{}/{}".format(i, len(filenames)))

        data_label = numpy.loadtxt(filename)
        data, label = block_xyz(
            data_label,
            6144,
            block_size=0.5,
            stride=0.5,
            random_sample=False,
            sample_num=None)

        label = numpy.array([label]).reshape((label.shape[0], label.shape[1], 1))
        data_label = numpy.concatenate([data, label], axis=2)
        basename = os.path.basename(filename)[:-4]

        if i < len(filenames) * 0.80:
            numpy.save("{}/{}.npy".format(train_folder, basename), data_label)
        else:
            numpy.save("{}/{}.npy".format(test_folder, basename), data_label)


if __name__ == "__main__":
    # organize_synthetic_data()
    # organize_floor_data()
    organize_data()