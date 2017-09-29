import cv2, caffe, os, time, sys
from caffe.proto import caffe_pb2
from PIL import Image
import numpy as np

###############################################################################
#                                I/O 					      #
###############################################################################

# parse arguments
if len(sys.argv) < 4:
	print '\nUsage: python inference.py <model> <weights> <working_dir>'
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
	print '\nUsage: python inference.py <model> <weights> <working_dir>'
	print '     : <working_dir> must contain at least 1 folder: image'
	print '                                           optional: disparity'
	sys.exit(1)
elif not os.path.exists(disparity_dir):
	print 'warning: no disparity directory found'

###############################################################################
#                                 SETUP                                       #
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

###############################################################################
#                                INFERENCE                                    #
###############################################################################

file_list = sorted(os.listdir(image_dir))
i = 0

while(True):
	filename = file_list[i].rsplit('_', 1)[0]
	print '\nProcessing: ' + filename

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
	time_start = time.time()
	output = net.forward()
	time_end = time.time()
	print 'inference time:', format(time_end - time_start, '.3f'), 's'

	#######################################
	#      MAP TO CITYSCAPES COLORS       #
	#######################################

	input_shape = net.blobs['data'].data.shape
	label_colours = cv2.imread(color_file, 1).astype(np.uint8)

	prediction = net.blobs['deconv6_0_0'].data[0].argmax(axis=0)
	prediction = np.squeeze(prediction)
	prediction = np.resize(prediction, (3, input_shape[2], input_shape[3]))
	prediction = prediction.transpose(1, 2, 0).astype(np.uint8)

	prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
	label_colours_bgr = label_colours[..., ::-1]
	cv2.LUT(prediction, label_colours_bgr, prediction_rgb)

	cv2.imshow("Prediction", prediction_rgb)
	key = cv2.waitKey(0)
	if key == 81:
		i = i - 1
	elif key == 82:
		continue
	elif key == 83:
		i = i + 1
	elif key == 27:
		sys.exit(1)
