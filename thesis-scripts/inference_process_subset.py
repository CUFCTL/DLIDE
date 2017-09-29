import cv2, caffe, os, sys
import numpy as np
from caffe.proto import caffe_pb2
from tqdm import tqdm
from PIL import Image

dataset_dir = '/media/eceftl9/data1/Datasets/Cityscapes'
image_dir = dataset_dir + '/leftImg8bit'
disparity_dir = dataset_dir + '/disparity8bit'
output_dir = dataset_dir + '/output'

###############################################################################
#									I/O 									  #
###############################################################################

# parse arguments
if len(sys.argv) < 4:
	print '\nUsage: python inference_process_subset.py <model> <weights> <subset>'
	print '       : <subset> must be either train, test, or val'
	sys.exit(1)

# environment setup
caffe_dir = os.getcwd()
model = os.path.join(caffe_dir,sys.argv[1])
weights = os.path.join(caffe_dir,sys.argv[2])
subset = sys.argv[3]
print 'caffe_dir = ', caffe_dir
print 'model = ', model
print 'weights = ', weights
print 'dataset_dir = ', dataset_dir
print 'image_dir = ', image_dir
print 'disparity_dir = ', disparity_dir
print 'output_dir = ', output_dir

# check if files exist
if not os.path.exists(model):
	print 'prototxt file does not exist'
	print 'looked for: ', model
	sys.exit(1)
elif not os.path.exists(weights):
	print 'weights file does not exist'
	print 'looked for: ', weights
	sys.exit(1)

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

###############################################################################
#								  INFERENCE 								  #
###############################################################################

for root, dirs, files in os.walk(image_dir + '/' + subset):
	if len(files) == 0:
		continue
	print root
	for filename in tqdm(files):
		name = filename.rsplit('_', 1)[0]

		#######################################
		#		 	LOAD AND COMPUTE          #
		#######################################

		for input_layer in range(0, num_inputs):
			if net.blobs[net.inputs[input_layer]].data.shape[1] == 3:
				img_rgb = Image.open(root + '/' + filename).convert('RGB')
				img_rgb = np.array(img_rgb)
				image_data = transformers[input_layer].preprocess('data', img_rgb)
				net.blobs[net.inputs[input_layer]].data[...] = image_data
			else:
				img_d = Image.open((root + '/' + filename).replace('leftImg8bit', 'disparity8bit')).convert('L')
				img_d = np.array(img_d)
				img_d = img_d[:, :, np.newaxis]
				image_data_d = transformers[input_layer].preprocess('data', img_d)
				net.blobs[net.inputs[input_layer]].data[...] = image_data_d

		# # compute
		output = net.forward()

		#######################################
		#       SAVE OUTPUT PREDICTION        #
		#######################################

		input_shape = net.blobs['data'].data.shape

		prediction = net.blobs['deconv6_0_0'].data[0].argmax(axis=0)
		prediction = np.squeeze(prediction).astype(np.uint8)

		out = Image.fromarray(prediction).convert('L')
		out.save(output_dir + '/' + name + '_predict.png')