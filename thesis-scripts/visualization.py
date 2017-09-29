import cv2, caffe, os, time, sys, math
import matplotlib.pyplot as plt
from PIL import Image
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

def show_viz(data):
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

    cv2.imshow("Blob", data)

###############################################################################
#									I/O 									  #
###############################################################################

# parse arguments
if len(sys.argv) < 4:
	print '\nUsage: python visualization.py <model> <weights> <working_dir>'
	print '     : <working_dir> must contain at least 1 folder: image'
	print '                                           optional: disparity'
	sys.exit(1)

# environment setup
caffe_dir = os.getcwd()
model = os.path.join(caffe_dir,sys.argv[1])
weights = os.path.join(caffe_dir,sys.argv[2])
image_dir = os.path.join(caffe_dir,sys.argv[3]) + 'image/'
disparity_dir = os.path.join(caffe_dir,sys.argv[3]) + 'disparity/'
color_file = caffe_dir + '/data/cityscapes19.png'
print 'caffe_dir = ', caffe_dir
print 'model = ', model
print 'weights = ', weights
print 'image_dir = ', image_dir
print 'disparity_dir = ', disparity_dir
print 'color_file = ', color_file

# check if files exist
if not os.path.exists(model):
	print 'prototxt file does not exist'
	print 'looked for: ', model
	sys.exit(1)
elif not os.path.exists(weights):
	print 'weights file does not exist'
	print 'looked for: ', weights
	sys.exit(1)
elif not os.path.exists(sys.argv[3]) or \
	 not os.path.exists(image_dir):
	print '\nUsage: python visualization.py <model> <weights> <working_dir>'
	print '     : <working_dir> must contain at least 1 folder: image'
	print '                                           optional: disparity'
	sys.exit(1)
elif not os.path.exists(disparity_dir):
	print 'warning: no disparity directory found'

###############################################################################
#									SETUP 									  #
###############################################################################

# create network
caffe.set_mode_gpu()
net = caffe.Net(model, weights, caffe.TRAIN)

# data transformers
num_inputs = len(net.inputs)
transformers = []
for input_layer in range(0, num_inputs):
	transformers.append(caffe.io.Transformer({'data':net.blobs[net.inputs[input_layer]].data.shape}))
	transformers[input_layer].set_transpose('data', (2,0,1))
	if net.blobs[net.inputs[input_layer]].data.shape[1] == 3:
		transformers[input_layer].set_channel_swap('data', (2,1,0))

###############################################################################
#                                INFERENCE                                    #
###############################################################################

file_list = sorted(os.listdir(image_dir))
i = 0
current_blob = []
while(True):
	filename = file_list[i].rsplit('_', 1)[0]

	#######################################
	#           LOAD AND COMPUTE          #
	#######################################

	for input_layer in range(0, num_inputs):
		if net.blobs[net.inputs[input_layer]].data.shape[1] == 3:
			img_rgb = Image.open(image_dir + filename + '_leftImg8bit.png').convert('RGB')
			img_rgb = np.array(img_rgb)
			image_data = transformers[input_layer].preprocess('data', img_rgb)
			net.blobs[net.inputs[input_layer]].data[...] = image_data
		else:
			img_d = Image.open(disparity_dir + filename + '_disparity8bit.png').convert('L')
			img_d = np.array(img_d)
			img_d = img_d[:, :, np.newaxis]
			image_data_d = transformers[input_layer].preprocess('data', img_d)
			net.blobs[net.inputs[input_layer]].data[...] = image_data_d

	# compute
	output = net.forward()

	########################################
	#               VIZUALIZE              #
	########################################

	# blobs
	blobs = [blob for blob in net.blobs.keys() if 'split' not in blob and 'mask' not in blob]
	if current_blob == []:
		blob_num = 0

	while(True):
		blob_name = blobs[blob_num]
		print filename, ' : ', blob_name
		current_blob = net.blobs[blob_name].data[0, :]
		current_blob = current_blob[:, np.newaxis, :, :]
		current_blob = np.repeat(current_blob, 3, axis=1).transpose(0, 2, 3, 1)
		show_viz(current_blob)
		key = cv2.waitKey(0)
		if key == 81:
			i = i - 1
			break
		elif key == 82:
			blob_num = blob_num + 1
		elif key == 83:
			i = i + 1
			break
		elif key == 84:
			blob_num = blob_num - 1
		elif key == 100:
			if '_D' in blob_name:
				blob_name = blob_name[:-2]
				blob_num = blobs.index(blob_name)
			else:
				if blob_name + '_D' in blobs:
					blob_num = blobs.index(blob_name + '_D')
				else:
					print 'No corresponding depth blob'
		elif key == 27:
			sys.exit(1)