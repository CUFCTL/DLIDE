#!/usr/bin/env python
"""
This script is a modified version of:
https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Scripts/compute_bn_statistics.py
For a description see: http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html

[1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing
internal covariate shift." arXiv preprint arXiv:1502.03167 (2015).
"""
import os, sys
from os.path import join

import numpy as np
from skimage.io import ImageCollection
from argparse import ArgumentParser

import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

caffe_root = '/home/eceftl9/jtetrea/MM-ENet/caffe-mm-enet/' # caffe-mm-enet directory
sys.path.insert(0, caffe_root + 'python')

################################################################################
################################################################################

def make_parser():
    p = ArgumentParser()
    p.add_argument('train_model')
    p.add_argument('weights')
    p.add_argument('out_dir')
    return p

################################################################################

def extract_DenseImageData(net_message):
    assert net_message.layer[0].type == "DenseImageData"
    source = net_message.layer[0].dense_image_data_param.source

    with open(source) as f:
        data = f.read().split()
    images = ImageCollection(data[::2])
    labels = ImageCollection(data[1::2])
    assert len(images) == len(labels) > 0

    return images, labels

def extract_DenseMMImageData(net_message):
    assert net_message.layer[0].type == "DenseMMImageData"
    source = net_message.layer[0].dense_mm_image_data_param.source

    with open(source) as f:
        data = f.read().split()
    images = ImageCollection(data[::3])
    d_images = ImageCollection(data[1::3])
    labels = ImageCollection(data[2::3])
    assert len(images) == len(labels) > 0
    assert len(images) == len(d_images) > 0

    return images, d_images, labels

################################################################################

def make_testable(train_model_path):
    # load the train net prototxt as a protobuf message
    with open(train_model_path) as f:
        train_str = f.read()
    train_net = caffe_pb2.NetParameter()
    text_format.Merge(train_str, train_net)

    # add the mean, var top blobs to all BN layers
    for layer in train_net.layer:
        if layer.type == "BN" and len(layer.top) == 1:
            layer.top.append(layer.top[0] + "-mean")
            layer.top.append(layer.top[0] + "-var")

    # remove the test data layer and 'include {phase: TRAIN}' parameter if present
    if "data" in train_net.layer[1].name and train_net.layer[1].include:
        train_net.layer.remove(train_net.layer[1])
        if train_net.layer[0].include:
            train_net.layer[0].include.remove(train_net.layer[0].include[0])

    return train_net

def make_test_files(testable_net_path, train_weights_path, num_iterations):
    # load the train net prototxt as a protobuf message
    with open(testable_net_path) as f:
        testable_str = f.read()
    testable_msg = caffe_pb2.NetParameter()
    text_format.Merge(testable_str, testable_msg)
    
    bn_layers = [l.name for l in testable_msg.layer if l.type == "BN"]
    bn_blobs = [l.top[0] for l in testable_msg.layer if l.type == "BN"]
    bn_means = [l.top[1] for l in testable_msg.layer if l.type == "BN"]
    bn_vars = [l.top[2] for l in testable_msg.layer if l.type == "BN"]
    conv_layers = [l.name for l in testable_msg.layer if l.type == "Convolution"]
    
    # init our blob stores with the first forward pass
    net = caffe.Net(testable_net_path, train_weights_path, caffe.TEST)
    res = net.forward()
    bn_avg_mean = {bn_mean: np.squeeze(res[bn_mean]).copy() for bn_mean in bn_means}
    bn_avg_var = {bn_var: np.squeeze(res[bn_var]).copy() for bn_var in bn_vars}

    # iterate over full of the training set
    for i in xrange(1, num_iterations):
        res = net.forward()
        for bn_mean in bn_means:
            bn_avg_mean[bn_mean] += np.squeeze(res[bn_mean])
        for bn_var in bn_vars:
            bn_avg_var[bn_var] += np.squeeze(res[bn_var])
        print 'Progress: {}/{}'.format(i, num_iterations)

    # compute average means and vars
    for bn_mean in bn_means:
        bn_avg_mean[bn_mean] /= num_iterations
    for bn_var in bn_vars:
        bn_avg_var[bn_var] /= num_iterations
    for bn_blob, bn_var in zip(bn_blobs, bn_vars):
        m = np.prod(net.blobs[bn_blob].data.shape) / np.prod(bn_avg_var[bn_var].shape)
        bn_avg_var[bn_var] *= (m / (m - 1))

    # calculate the new scale and shift blobs for all the BN layers
    scale_data = {bn_layer: np.squeeze(net.params[bn_layer][0].data) for bn_layer in bn_layers}
    shift_data = {bn_layer: np.squeeze(net.params[bn_layer][1].data) for bn_layer in bn_layers}
    new_scale_data = {}
    new_shift_data = {}

    var_eps = 1e-9
    for bn_layer, bn_mean, bn_var in zip(bn_layers, bn_means, bn_vars):
        gamma = scale_data[bn_layer]
        beta = shift_data[bn_layer]
        Ex = bn_avg_mean[bn_mean]
        Varx = bn_avg_var[bn_var]

        new_gamma = gamma / np.sqrt(Varx + var_eps)
        new_beta = beta - (gamma * Ex / np.sqrt(Varx + var_eps))

        new_scale_data[bn_layer] = new_gamma
        new_shift_data[bn_layer] = new_beta
    print "New data:", new_scale_data.keys(), new_shift_data.keys()

    # assign computed new scale and shift values to net.params
    for bn_layer in bn_layers:
        net.params[bn_layer][0].data[...] = new_scale_data[bn_layer].reshape(
            net.params[bn_layer][0].data.shape)
        net.params[bn_layer][1].data[...] = new_shift_data[bn_layer].reshape(
            net.params[bn_layer][1].data.shape)

    # build a test net prototxt
    test_msg = testable_msg

    return net, test_msg, new_scale_data, new_shift_data, conv_layers


def make_deployable(net_message, in_h, in_w):
    deploy_msg = net_message

    # replace data layers with 'input' net param
    data_layers = [l for l in deploy_msg.layer if l.type.endswith("Data")]
    for data_layer in data_layers:
        deploy_msg.layer.remove(data_layer)

    deploy_msg.input.append("data")
    deploy_msg.input_dim.append(1)
    deploy_msg.input_dim.append(3)
    deploy_msg.input_dim.append(in_h)
    deploy_msg.input_dim.append(in_w)

    deploy_msg.input.append("depth")
    deploy_msg.input_dim.append(1)
    deploy_msg.input_dim.append(1)
    deploy_msg.input_dim.append(in_h)
    deploy_msg.input_dim.append(in_w)

    # set BN layers to INFERENCE and remove mean and var top blobs
    for l in deploy_msg.layer:
        if l.type == "BN":
            if len(l.top) > 1:
                dead_tops = l.top[1:]
                for dl in dead_tops:
                    l.top.remove(dl)
            l.bn_param.bn_mode = caffe_pb2.BNParameter.INFERENCE

    # replace output loss, accuracy layers with a softmax
    dead_outputs = [l for l in deploy_msg.layer if l.type in ["SoftmaxWithLoss", "Accuracy"]]
    out_bottom = dead_outputs[0].bottom[0]
    for dead in dead_outputs:
        deploy_msg.layer.remove(dead)
    deploy_msg.layer.add(name="prob", type="Softmax", bottom=[out_bottom], top=['prob'])

    return deploy_msg

################################################################################
################################################################################

if __name__ == '__main__':
    caffe.set_mode_gpu()

    p = make_parser()
    args = p.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # build and save testable net
    print "Building BN calc net..."
    batch_size = 2  # change batch if needed size 
    testable_msg = make_testable(args.train_model)
    if "MM" in args.train_model:
        train_images, train_d_images, train_labels = extract_DenseMMImageData(testable_msg)
        in_h = testable_msg.layer[0].dense_mm_image_data_param.new_height
        in_w = testable_msg.layer[0].dense_mm_image_data_param.new_width
        testable_msg.layer[0].dense_mm_image_data_param.batch_size = batch_size
    else:
        train_images, train_labels = extract_DenseImageData(testable_msg)
        in_h = testable_msg.layer[0].dense_image_data_param.new_height
        in_w = testable_msg.layer[0].dense_image_data_param.new_width
        testable_msg.layer[0].dense_image_data_param.batch_size = batch_size

    BN_calc_path = os.path.join(args.out_dir, '__for_calculating_BN_stats_' + os.path.basename(args.train_model))
    with open(BN_calc_path, 'w') as f:
        f.write(text_format.MessageToString(testable_msg))

    # get network input info and update batch size
    print "Calculating BN layers.stat..."
    num_iterations = len(train_images) // batch_size + len(train_images) % batch_size
    test_net, test_msg, new_gamma, new_beta, conv_layers = make_test_files(BN_calc_path, args.weights, num_iterations)
    os.remove(BN_calc_path)

    print "Generating deployment files..."
    deploy_msg = make_deployable(test_msg, in_h, in_w)
    test_path = os.path.join(args.out_dir, "deploy.prototxt")
    with open(test_path, 'w') as f:
        f.write(text_format.MessageToString(deploy_msg))

    print "Saving test net weights..."
    test_net.save(os.path.join(args.out_dir, "weights_bn.caffemodel"))
    print "Done"