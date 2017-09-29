import cv2, caffe, os, time, sys, math
import matplotlib.pyplot as plt
import scipy.misc, PIL.Image
from caffe.proto import caffe_pb2
import numpy as np

###############################################################################
#							   HELPER FUNCTION								  #
###############################################################################

def save_viz(data, output):
    '''Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. 
       sqrt(n) by sqrt(n)'''
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n**2 - data.shape[0]), (0, 1), (0, 1))    	  	  # add some space between filters
            + ((0, 0),) * (data.ndim - 3))  						  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imsave(output, data)

###############################################################################
#									I/O 									  #
###############################################################################

# parse arguments
if len(sys.argv) < 5:
	print '\nUsage: python visualization.py <model> <weights> <filename> <working_dir>'
	print '     : <working_dir> must contain at least 3 folders: image, filters, blobs, disparity (optional)'
	sys.exit(1)

# initialize environment
caffe_dir = os.getcwd()
model = os.path.join(caffe_dir,sys.argv[1])
weights = os.path.join(caffe_dir,sys.argv[2])
filename = os.path.join(caffe_dir,sys.argv[3])
image_dir = os.path.join(caffe_dir,sys.argv[4]) + 'image/'
filter_dir = os.path.join(caffe_dir,sys.argv[4]) + 'filters/'
blob_dir = os.path.join(caffe_dir,sys.argv[4]) + 'blobs/'
disparity_dir = os.path.join(caffe_dir,sys.argv[4]) + 'disparity/'

print '\ncaffe_dir = ', caffe_dir
print 'prototxt = ', model
print 'weights = ', weights
print 'filename = ', filename
print 'filter_dir = ', filter_dir
print 'blob_dir = ', blob_dir
print 'disparity_dir = ', disparity_dir

# check if files exist
if not os.path.exists(model):
	print 'model file does not exist'
	sys.exit(1)
elif not os.path.exists(weights):
	print 'weights file does not exist'
	sys.exit(1)
elif not os.path.exists(filename):
	print 'image does not exist'
	sys.exit(1)
elif not os.path.exists(sys.argv[4]) or \
	 not os.path.exists(filter_dir) or \
	 not os.path.exists(blob_dir):
	print '\n<working_dir> structure incorrect'
	print '<working_dir> must contain at least 3 folders: image, filters, blobs, disparity (optional)'
	sys.exit(1)
elif not os.path.exists(disparity_dir):
	print 'warning: no disparity directory found'

###############################################################################
#									SETUP 									  #
###############################################################################

# create network
caffe.set_mode_gpu()
net = caffe.Net(model, weights, caffe.TEST)

# data transformers
num_inputs = len(net.inputs)
transformers = []
for input_layer in range(0, num_inputs):
	transformers.append(caffe.io.Transformer({'data':net.blobs[net.inputs[input_layer]].data.shape}))
	transformers[input_layer].set_transpose('data', (2,0,1))
	if net.blobs[net.inputs[input_layer]].data.shape[1] == 3:
		transformers[input_layer].set_channel_swap('data', (2,1,0))

filename = os.path.basename(filename).rsplit('_', 1)[0]
print filename

for input_layer in range(0, num_inputs):
	if net.blobs[net.inputs[input_layer]].data.shape[1] == 3:
		img_rgb = PIL.Image.open(image_dir + filename + '_leftImg8bit.png').convert('RGB')
		img_rgb = np.array(img_rgb)
		image_data = transformers[input_layer].preprocess('data', img_rgb)
		net.blobs[net.inputs[input_layer]].data[...] = image_data
	else:
		img_d = PIL.Image.open(disparity_dir + filename + '_disparity.png').convert('L')
		img_d = np.array(img_d)
		img_d = img_d[:, :, np.newaxis]
		image_data_d = transformers[input_layer].preprocess('data', img_d)
		net.blobs[net.inputs[input_layer]].data[...] = image_data_d

# compute
output = net.forward()

###############################################################################
#								  VIZUALIZE    								  #
###############################################################################

# paramaters
print '\nVisualizing parameters...'
for layer_name in net.params.keys():
	if 'conv' in layer_name or 'inception' in layer_name \
	and 'reduce' not in layer_name and 'proj' not in layer_name:
		filters = net.params[layer_name][0].data
		print filters.shape, layer_name
		if filters.shape[1] > 3:
			filters = np.mean(filters, axis = 1)
			# filters = filters[:, np.newaxis, :, :]
			# filters = np.repeat(filters, 3, axis=1).transpose(0, 2, 3, 1)
			save_viz(filters, filter_dir + (layer_name).replace('/','_'))
		else:
			if filters.shape[1] == 1:
				filters = np.repeat(filters, 3, axis=1)
			save_viz(filters.transpose(0, 2, 3, 1), filter_dir + (layer_name).replace('/','_'))

# blobs
print '\nVisualizing blobs...'
for blob_name in net.blobs.keys():
	if 'conv' in blob_name or 'inception' in blob_name \
	and 'reduce' not in blob_name and 'proj' not in blob_name \
	and 'split' not in blob_name:
		blobs = net.blobs[blob_name].data[0, :]
		print blobs.shape, blob_name
		blobs = blobs[:, np.newaxis, :, :]
		blobs = np.repeat(blobs, 3, axis=1).transpose(0, 2, 3, 1)
		save_viz(blobs, blob_dir + (blob_name).replace('/','_'))
