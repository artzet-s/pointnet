import indoor3d_util
import numpy
import glob
import os

input_dir = "/home/artzet_s/code/dataset/afef_apple_tree_filtred_labeled/aug"
filenames = glob.glob(os.path.join(input_dir, '*.txt'))
filename = filenames[0]

print(filename)
pc = numpy.loadtxt(filename)


print('split')
indices = indoor3d_util.split_3d_point_cloud_to_several_windows(pc,
                                                                (0.5, 0.5, 0.5))
v = numpy.ones_like(pc[:, 0])
print(pc.shape,  v.shape)

for i, cond in enumerate(indices):

    if numpy.count_nonzero(cond) < 1000:
        continue

    if numpy.count_nonzero(cond) < 1000:
        v[cond] = 0
    elif numpy.count_nonzero(cond) < 4000:
        v[cond] = 10
    elif numpy.count_nonzero(cond) < 8000:
        v[cond] = 20
    else:
        v[cond] = 30
res = numpy.column_stack([pc, v])
numpy.savetxt("test.txt", res)


