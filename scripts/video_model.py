from utils import *
from vgg_model import *
import tensorflow as tf

def temporal_encoder(inputs):
    with tf.variable_scope('encoder3d_1'):
        conv1_1 = conv3d_layer(inputs, 3, 64 ,"conv1_1")
        pool1 = max_pool3d1(conv1_1, 'pool1')

    with tf.variable_scope('encoder3d_2'):
        conv2_1 = conv3d_layer(pool1, 64, 128, "conv2_1")
        norm = tf.contrib.slim.batch_norm(conv2_1)
        pool2 = max_pool3d(norm, 'pool2')

    with tf.variable_scope('encoder3d_3'):
        conv3_1 = conv3d_layer(pool2, 128, 256, "conv3_1")
        conv3_2 = conv3d_layer(conv3_1, 256, 256, "conv3_2")
        norm = tf.contrib.slim.batch_norm(conv3_2)
        pool3 = max_pool3d(norm, 'pool3')

    with tf.variable_scope('encoder3d_4'):
        conv4_1 = conv_layer(pool3, 256, 512, "conv4_1")
        conv4_2 = conv_layer(conv4_1, 512, 512, "conv4_2")
	norm = tf.contrib.slim.batch_norm(conv4_2)
	pool4 = max_pool3d(conv4_1,'pool4')

    with tf.variable_scope('encoder3d_5'):
        conv5_1 = conv_layer(pool4, 512, 512, "conv5_1")
        conv5_2 = conv_layer(conv4_1, 512, 512, "conv5_2")
        norm = tf.contrib.slim.batch_norm(conv5_2)
        pool5 = max_pool3d(conv4_1,'pool5')
    layers = [pool1,pool2,pool3,pool4,pool5]
    return layers
    
def combine_encodings(vgg, temporal, past):
    with tf.variable_scope('concatinate'):
	combined = tf.concat(tf.concat(vgg,temporal,axis=4),past,axis = 4)

    with tf.variable_scope('weight_generator'):
	conv6_1 = conv_layer({},combined,512*3,512,"conv6_1")
        fc_input = tf.contrib.layers.flatten(conv6_1) 
        fc6_1 = tf.contrib.slim.fully_connected(fc_input,512*3,scope = "fc6_1")
	out6 = 512 * tf.nn.softmax(fc7)
    
    for i in range(512):
	vgg[:,:,:,i] = vgg[:,:,:,i]*out6[:i]+temporal[:,:,:,i]*out6[:,i+512*1]+past[:,:,:,i]*out[:,i*512*2]
    
    return vgg

 
