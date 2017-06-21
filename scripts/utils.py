
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import os
import glob
import random
import collections
import math
import time
import sys
import re



def nss_calc(gtsAnn, resAnn):

    salMap = (resAnn - np.mean(resAnn))/255.0
    if np.max(salMap) > 0:
	salMap = salMap / np.std(salMap)
    gtsAnn = np.sign(gtsAnn/255.0)
    a = np.multiply(gtsAnn,resAnn)
    
    return np.sum(a)/np.count_nonzero(a)



def kld(pred,target):
    q = pred
    q /= np.sum(pred.flatten())
    p = target
    p/= np.sum(p.flatten())
    q+=np.finfo(q.dtype).eps
    p+=np.finfo(p.dtype).eps
    kl = np.sum(p*(np.log2(p/q)))
    return kl
 
    



def shuffled(x):
    y = x[:]
    random.shuffle(y)
    return y


def save_image (image,output_dir,filename):
    import cv2
    cv2.imwrite(output_dir+filename+'.png', image)
    return


def save_image2 (image,output_dir,filename):
    
    plt.imsave(output_dir+filename+'.png', image)
    return


class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)





def conv3d_layer(inputs,in_channels,out_channels,name):
    with tf.variable_scope(name):
	filter_size = 3
        filt = tf.get_variable(name+'_filter', [filter_size, filter_size, filter_size, in_channels, out_channels], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev=0.1))
	biases = tf.get_variable(name+'_bias',[out_channels],tf.float32, tf.constant_initializer(0.1, dtype=tf.float32))
        conv = tf.nn.conv3d(inputs, filt, [1, 1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        relu = lrelu(bias,0.2)
        return relu



def max_pool3d(input,name):
    return tf.nn.max_pool3d(input, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME',name = name)

def max_pool3d1(input,name):
    return tf.nn.max_pool3d(input, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME',name = name)



def get_var(data_dict,initial_value, name, idx, var_name):
    if data_dict is not None and name in data_dict:
        value = data_dict[name[-7:]][idx]
    else:
	value = initial_value
    var = tf.Variable(value, name=var_name)
    return var




def get_conv_var(data_dict, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = get_var(data_dict,initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = get_var(data_dict,initial_value, name, 1, name + "_biases")

        return filters, biases

def conv_layer(data_dict,bottom, in_channels, out_channels, name):
    with tf.variable_scope(name):
	filt, conv_biases = get_conv_var(data_dict,3, in_channels, out_channels, name)

	conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
	bias = tf.nn.bias_add(conv, conv_biases)
	relu = lrelu(bias,0.2)

	return relu


def get_var2(data_dict,initial_value, name,  var_name):
    for key in data_dict.keys():
        if data_dict is not None and var_name in key:
            value = data_dict[key]
    else:
        value = initial_value
    var = tf.Variable(value, name=var_name)
    return var


def get_conv_var2(data_dict, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = get_var(data_dict,initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = get_var(data_dict,initial_value, name, 1, name + "_biases")

        return filters, biases

def conv_layer2(data_dict,bottom, in_channels, out_channels, name):
    with tf.variable_scope(name):
        filt, conv_biases = get_conv_var(data_dict,3, in_channels, out_channels, name)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        relu = lrelu(bias,0.2)

        return relu

def max_pool( bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)





def conv(batch_input, out_channels, stride):
    """Convoluton with filter size 4 x 4 x in_channels x out_channels 
	and padding valid"""
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

def conv11(batch_input,out_channels=1, stride=1):

    with tf.variable_scope("conv11"):
        in_channels = batch_input.get_shape()[3]
	
        filter = tf.get_variable("filter", [1, 1, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        #padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(batch_input, filter, [1, stride, stride, 1], padding="SAME")
        return conv

def conv112(batch_input,data_dict = {} ,out_channels=1,in_channels=64, stride=1):

    with tf.variable_scope("conv11"):
	
        initial_filter = tf.truncated_normal( [1, 1, in_channels, out_channels], 0, 0.02)
	filter = get_var2(data_dict,initial_filter,'conv11','conv11/filter')
        conv = tf.nn.conv2d(batch_input, filter, [1, stride, stride, 1], padding="SAME")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
   input = tf.identity(input)

   channels = input.get_shape()[3]
   offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
   scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
   mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
   variance_epsilon = 1e-5
   normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)

def deconv(batch_input, out_channels,filter_size = 4):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [filter_size, filter_size, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv

def deconv_layer(batch_input, out_channels,filter_size,stride,name):
    with tf.variable_scope(name):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [filter_size, filter_size, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * stride, in_width * stride, out_channels], [1, stride, stride, 1], padding="SAME")
        bias = tf.truncated_normal([out_channels], .0, .001)
	conv = tf.nn.relu(tf.nn.bias_add(conv,bias))
        return conv





def deconv_layer2(batch_input, out_channels,filter_size,stride,name,data_dict):
    with tf.variable_scope(name):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        initial_filter = tf.random_normal([filter_size, filter_size, out_channels, in_channels],0, 0.02)
        filter = get_var2(data_dict,initial_filter,name,name + '/filter')
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * stride, in_width * stride, out_channels], [1, stride, stride, 1], padding="SAME")
        initial_bias = tf.truncated_normal([out_channels], .0, .001)
	bias = get_var2(data_dict,initial_bias,name,name+'/bias')
        conv = tf.nn.relu(tf.nn.bias_add(conv,bias))
        return conv









def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))

def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))

