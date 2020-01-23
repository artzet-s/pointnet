import glob
import numpy


input_folder = "/home/artzet_s/code/dataset/afef_apple_tree_filtred"


vmax, vmin = list(), list()
for i, filename in enumerate(glob.glob("{}/pc_2018*.txt".format(input_folder))):
	data_label = numpy.loadtxt(filename)
	vmax.append(numpy.amax(data_label, axis=0))
	vmin.append(numpy.amin(data_label, axis=0))
vmax = numpy.amax(numpy.array(vmax), axis=0)
vmin = numpy.amax(numpy.array(vmin), axis=0)

arr = numpy.stack([vmin, vmax], axis=0)
print(arr)
numpy.savetxt("mean_data.txt", arr)
