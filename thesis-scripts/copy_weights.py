import cv2, caffe, os, time, sys, math
import matplotlib.pyplot as plt
import scipy.misc, PIL.Image
from caffe.proto import caffe_pb2
import numpy as np
from caffe import layers as L, params as P

def rgb2gray(rgb):

    r, g, b = rgb[:,0,:,:], rgb[:,1,:,:], rgb[:,2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    print gray[:,np.newaxis,:,:].shape
    return gray[:,np.newaxis,:,:]

###############################################################################
#									I/O 									  #
###############################################################################

# parse arguments
if len(sys.argv) < 4:
	print 'Usage: python copy_weights.py <model> <new_model> <weights>'
	sys.exit(1)

# initialize environment
caffe_dir = os.getcwd()
model = os.path.join(caffe_dir,sys.argv[1])
new_model = os.path.join(caffe_dir,sys.argv[2])
weights = os.path.join(caffe_dir,sys.argv[3])
print '\ncaffe_dir = ', caffe_dir
print 'model = ', model
print 'new_model = ', new_model
print 'weights = ', weights

# check if files exist
if not os.path.exists(model):
	print 'prototxt file does not exist'
	print 'looked for: ', model
	sys.exit(1)
elif not os.path.exists(new_model):
	print 'new_model file does not exist'
	print 'looked for: ', new_model
	sys.exit(1)
elif not os.path.exists(weights):
	print 'weights file does not exist'
	print 'looked for: ', weights
	sys.exit(1)

###############################################################################
#									SETUP 									  #
###############################################################################
caffe.set_mode_gpu()

# create network
net = caffe.Net(model, weights, caffe.TEST)
new_net = caffe.Net(new_model, caffe.TEST)
full_net = caffe.Net(new_model, caffe.TEST)

###############################################################################
#							   COPY PARAMETERS 								  #
###############################################################################

# types of copies
new_weights = 'new_' + os.path.basename(os.path.normpath(weights))
full_weights = 'full_' + os.path.basename(os.path.normpath(weights))

for layer_name in new_net.params.keys():
	if '_D' in layer_name:
		print 'copying', layer_name, 'to full_net'
		if net.params[layer_name[:-2]][0].data.shape[1] == full_net.params[layer_name][0].data.shape[1]*3:
			print net.params[layer_name[:-2]][0].data[...].shape
			full_net.params[layer_name][0].data[...] = rgb2gray(net.params[layer_name[:-2]][0].data[...])
		else:
			full_net.params[layer_name][0].data[...] = net.params[layer_name[:-2]][0].data[...]
	if layer_name in net.params.keys():
		# double appropriate parameters for new modality
		if net.params[layer_name][0].data.shape[1] == new_net.params[layer_name][0].data.shape[1]/2:
			print 'doubling', layer_name
			new_net.params[layer_name][0].data[...] = np.repeat(net.params[layer_name][0].data[...], 2, axis = 1)
			full_net.params[layer_name][0].data[...] = np.repeat(net.params[layer_name][0].data[...], 2, axis = 1)
		# convert from 2 class network to 1 class
		elif 'cvg/classifier' in layer_name:
			print 'copying first class from', layer_name
			new_net.params[layer_name][0].data[...] = net.params[layer_name][0].data[0, :, :, :]
			full_net.params[layer_name][0].data[...] = net.params[layer_name][0].data[0, :, :, :]
		else:
			print 'copying', layer_name
			new_net.params[layer_name][0].data[...] = net.params[layer_name][0].data[...]
			full_net.params[layer_name][0].data[...] = net.params[layer_name][0].data[...]

new_net.save(os.path.join(os.path.dirname(weights),new_weights))
full_net.save(os.path.join(os.path.dirname(weights),full_weights))