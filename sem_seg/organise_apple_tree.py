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
            # randomly subsample data
            block_data_sampled, block_label_sampled = indoor3d_util.sample_data_label(
                block_data, block_label, num_point)

            # Keep point if number of apple point are superior to 20
            if numpy.count_nonzero(block_label_sampled) > 50:
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

    data = data_label[:, 0:3]
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


def organize_data():

    input_folders = [
        "/home/artzet_s/code/dataset/afef_apple_tree_filtred_labeled/aug_train",
        "/home/artzet_s/code/dataset/afef_apple_tree_filtred_labeled/aug_test"]

    output_folders = [
        "/home/artzet_s/code/dataset/train_block_data",
        "/home/artzet_s/code/dataset/test_block_data"]

    min_max = numpy.loadtxt("mean_data.txt")

    # for input_folders, output_folder in zip(input_folders, output_folders):

    #     if not os.path.exists(output_folder):
    #         os.mkdir(output_folder)

    #     filenames = glob.glob(os.path.join(input_folders, "*.npy"))
    #     for i, filename in enumerate(filenames):

    #         basename = os.path.basename(filename)[:-4]
    #         output_filename = os.path.join(output_folder,
    #                                        "{}.npy".format(basename))

    #         if not os.path.exists(output_filename):
    #             compute_block(filename, output_filename, min_max)
            
    #         print("{}/{} {}".format(i, len(filenames), output_filename))

    elements = list()
    for input_folders, output_folder in zip(input_folders, output_folders):

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        filenames = glob.glob(os.path.join(input_folders, "*.npy"))
        for i, filename in enumerate(filenames):

            basename = os.path.basename(filename)[:-4]
            output_filename = os.path.join(output_folder,
                                           "{}.npy".format(basename))

            if not os.path.exists(output_filename):
                elements.append((filename, output_filename, min_max.copy()))

    print(elements)
    nb_process = 4
    pool = multiprocessing.Pool(nb_process)
    pool.starmap(compute_block, elements)


def organize_light_data():

    input_folders = [
        "/home/artzet_s/code/dataset/afef_apple_tree_filtred_labeled/aug_train",
        "/home/artzet_s/code/dataset/afef_apple_tree_filtred_labeled/aug_test"]

    output_folders = [
        "/home/artzet_s/code/dataset/train_light_block_data",
        "/home/artzet_s/code/dataset/test_light_block_data"]

    min_max = numpy.loadtxt("mean_data.txt")

    for input_folders, output_folder in zip(input_folders, output_folders):

        initialize_folder(output_folder)
        filenames = glob.glob(os.path.join(input_folders, "*.npy"))

        for i, filename in enumerate(filenames):
            basename = os.path.basename(filename)[:-4]
            print("{}, {}/{}".format(filename, i, len(filenames)))

            data_label = numpy.load(filename)
            data, label = block_xyzrad(
                data_label,
                4096,
                min_max)

            label = numpy.array([label]).reshape(
                (label.shape[0], label.shape[1], 1))

            # Keep only the block with enought apple point
            new_data, new_label = list(), list()
            for i in range(data.shape[0]):
                if numpy.count_nonzero(label[i, :]) > 0:
                    new_data.append(data[i, ...])
                    new_label.append(label[i, :])

            if not new_data:
                continue

            data = numpy.stack(new_data)
            label = numpy.stack(new_label)

            data_label = numpy.concatenate([data, label], axis=2)

            numpy.save(os.path.join(output_folder, "{}.npy".format(basename)),
                       data_label)


def organize_synthetic_data():

    input_folder = "/home/artzet_s/code/dataset/synthetic_lidar_simulation"
    train_folder = "/home/artzet_s/code/dataset/train_block_synthetic_data"
    test_folder = "/home/artzet_s/code/dataset/test_block_synthetic_data"

    initialize_folder(train_folder)
    initialize_folder(test_folder)

    filenames = glob.glob("{}/*.txt".format(input_folder))
    random.shuffle(filenames)

    for i, filename in enumerate(filenames):
        basename = os.path.basename(filename)[:-4]
        print("{} {}/{}".format(basename, i, len(filenames)))

        data_label = numpy.loadtxt(filename)
        data, label = block_xyz(data_label, 4096)

        label = numpy.array([label]).reshape((label.shape[0], label.shape[1], 1))
        data_label = numpy.concatenate([data, label], axis=2)

        if i < len(filenames) * 0.80:
            numpy.save("{}/{}.npy".format(train_folder, basename), data_label)
        else:
            numpy.save("{}/{}.npy".format(test_folder, basename), data_label)


if __name__ == "__main__":
    # organize_synthetic_data()
    organize_data()
    # organize_light_data()
