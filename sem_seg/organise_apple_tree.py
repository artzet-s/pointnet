import os
import numpy
import sys
import glob
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import shutil
import random
import multiprocessing
import indoor3d_util

def multiprocess_function(function, elements, nb_process=2):
    pool = multiprocessing.Pool(nb_process)

    nb_elements = len(elements)

    it = pool.imap_unordered(function, elements)
    
    for i in range(nb_elements):
        try:
            it.next()
            
            print("{} : {} / {} ".format(function, i, nb_elements))
            sys.stdout.flush()

        except Exception as e:
            print("{} : {} / {} - ERROR {}".format(
				function, i, nb_elements, e))
            sys.stdout.flush()
    pool.close()
    pool.join()
    print("%s : %d / %d" % (function, nb_elements, nb_elements))
    sys.stdout.flush()


def _get_indice_3d_windows(xyz, x0, xn, y0, yn, z0, zn):
    indx = numpy.bitwise_and(x0 <= xyz[:, 0], xyz[:, 0] < xn)
    indy = numpy.bitwise_and(y0 <= xyz[:, 1], xyz[:, 1] < yn)
    indz = numpy.bitwise_and(z0 <= xyz[:, 2], xyz[:, 2] < zn)

    return numpy.bitwise_and(numpy.bitwise_and(indx, indy), indz)


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

    window_size=(0.25, 0.25, 0.25)
    # Collect blocks

    block_data_list = []
    block_label_list = []

    xyz = data[:, :3]
    ws = numpy.array(window_size)
    xyz_max = numpy.max(xyz, axis=0)
    xyz_min = numpy.min(xyz, axis=0)
    pc_nb = numpy.ceil((xyz_max - xyz_min) / ws).astype(int)

    for i, j, k in numpy.ndindex((pc_nb[0], pc_nb[1], pc_nb[2])):

        x0, y0, z0 = xyz_min + ws * numpy.array([i, j, k])
        xn, yn, zn = xyz_min + ws * numpy.array([i + 1, j + 1, k + 1])

        cond = _get_indice_3d_windows(xyz, x0, xn, y0, yn, z0, zn)
    
        if numpy.count_nonzero(cond) < 500:
            continue

        block_data, block_label = data[cond], label[cond]
    
        block_data_sampled, block_label_sampled = indoor3d_util.room2samples(
            block_data, block_label, num_point, K=K)

        if test_mode:
            for i in range(block_data_sampled.shape[0]):
                block_data_list.append(
                    numpy.expand_dims(block_data_sampled[i, :, ], 0))
                block_label_list.append(
                    numpy.expand_dims(block_label_sampled[i, :, 0], 0))
        else:
            if numpy.count_nonzero(block_label) > 100 and numpy.count_nonzero(block_label == 0) > 100:

                indice_noise = numpy.random.choice(numpy.where(block_label == 0)[0], num_point // 2)
                indice_apple = numpy.random.choice(numpy.where(block_label == 1)[0], num_point // 2)


                block_data_sampled = numpy.concatenate([block_data[indice_noise, ...], 
                                                        block_data[indice_apple, ...]])

                block_label_sampled = numpy.concatenate([block_label[indice_noise, ...], 
                                                         block_label[indice_apple, ...]])

                                                                                                                                           
                block_data_list.append(numpy.expand_dims(block_data_sampled, 0))
                block_label_list.append(numpy.expand_dims(block_label_sampled, 0))

    if block_data_list:
        return numpy.concatenate(block_data_list, 0), numpy.concatenate(block_label_list, 0)
    else:
        return None, None


def block_xyzrad(data_label,
                 num_point,
                 min_max,
                 test_mode=False):

    data = data_label[:, :6]
    label = data_label[:, -1].astype(numpy.uint8)

    # CENTRALIZE HERE
    data[:, :3] = data[:, :3] - numpy.amin(data, 0)[0:3]

    # Normalize Attribute value
    data[:, 3:6] = (data[:, 3:6] - min_max[0, 3:6]) / min_max[1, 3:6]

    data_batch, label_batch = build_blocks(data,
                                          label,
                                          num_point,
                                          test_mode,
                                           K=6)

    return data_batch, label_batch


def block_xyz(data_label,
              num_point,
              test_mode=False):

    data = data_label[:, :3]
    label = data_label[:, -1].astype(numpy.uint8)

    # CENTRALIZE HERE
    data[:, :3] = data[:, :3] - numpy.amin(data, 0)[0:3]


    data_batch, label_batch = build_blocks(data,
                                          label,
                                          num_point,
                                          test_mode,
                                           K=3)

    return data_batch, label_batch


def compute_block(input_filename, output_filename, min_max):

    print(input_filename)
    data_label = numpy.load(input_filename)
    data, label = block_xyzrad(
        data_label,
        4096,
        min_max)
    
    if data is not None:
        label = numpy.array([label]).reshape((label.shape[0], label.shape[1], 1))
        label = numpy.array([label]).reshape((label.shape[0], label.shape[1], 1))
        data_label = numpy.concatenate([data, label], axis=2)
        numpy.save(output_filename, data_label)


def compute_xyz_block(input_filename, output_filename):

    data_label = numpy.loadtxt(input_filename)
    data, label = block_xyz(
        data_label,
        4096)
    
    if data is not None:
        label = numpy.array([label]).reshape((label.shape[0], label.shape[1], 1))
        label = numpy.array([label]).reshape((label.shape[0], label.shape[1], 1))
        data_label = numpy.concatenate([data, label], axis=2)
        numpy.save(output_filename, data_label)


def organize_data():

    # input_folders = [
    #     "/home/artzet_s/code/dataset/afef_apple_tree_filtred_labeled/train",
    #     "/home/artzet_s/code/dataset/afef_apple_tree_filtred_labeled/test"]

    # output_folders = [
    #     "/home/artzet_s/code/dataset/pn_train_xyzadr_25cm3_balanced",
    #     "/home/artzet_s/code/dataset/pn_test_xyzadr_25cm3_balanced"]


    input_folders = [
        "/home/artzet_s/code/dataset/afef_apple_tree_filtred_labeled/aug_train",
        "/home/artzet_s/code/dataset/afef_apple_tree_filtred_labeled/aug_test"]

    output_folders = [
        "/home/artzet_s/code/dataset/pn_train_aug_xyzadr_25cm3_balanced",
        "/home/artzet_s/code/dataset/pn_test_aug_xyzadr_25cm3_balanced"]


    # input_folders = [
    #     "/home/artzet_s/code/dataset/synthetic_data/train",
    #     "/home/artzet_s/code/dataset/synthetic_data/test"]

    # output_folders = [
    #     "/home/artzet_s/code/dataset/synthetic_data/train_block_synthetic_data",
    #     "/home/artzet_s/code/dataset/synthetic_data/test_block_synthetic_data"]


    min_max = numpy.loadtxt("mean_data.txt")

    elements = list()
    for input_folders, output_folder in zip(input_folders, output_folders):

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        filenames = glob.glob(os.path.join(input_folders, "*.npy"))
        # filenames = glob.glob(os.path.join(input_folders, "*.txt"))
        for i, filename in enumerate(filenames):

            basename = os.path.basename(filename)[:-4]
            output_filename = os.path.join(output_folder,
                                           "{}.npy".format(basename))

            if not os.path.exists(output_filename):
                elements.append((filename, output_filename, min_max.copy()))
                # elements.append((filename, output_filename))

    print(len(elements))
    print(elements)
    nb_process = 4
    pool = multiprocessing.Pool(nb_process)
    pool.starmap(compute_block, elements)
    # pool.starmap(compute_xyz_block, elements)


if __name__ == "__main__":
    organize_data()
