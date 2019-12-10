#
#
#           RE-FORMAT DATA - TXT TO NPY
#
#
# ==============================================================================
import os
import sys
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

import indoor3d_util
# ==============================================================================


anno_paths = [line.rstrip() for line in open(
    os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
anno_paths = [os.path.join(indoor3d_util.DATA_PATH, p) for p in anno_paths]

output_folder = os.path.join(ROOT_DIR, 'data/stanford_indoor3d') 
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# WARNNG ! Note:
# There is an extra character in the v1.2 data in Area_5/hallway_6.
# It must be fixed manually.
for anno_path in anno_paths:
    elements = anno_path.split('/')

    # Area_1_hallway_1.npy
    out_filename = "{}/{}_{}.npy".format(
        output_folder, elements[-3], elements[-2])

    if not os.path.exists(out_filename):
        try:
            indoor3d_util.collect_point_label(
                anno_path, out_filename, 'numpy')
        except Exception as e:
            print(anno_path, 'ERROR!!', e)
