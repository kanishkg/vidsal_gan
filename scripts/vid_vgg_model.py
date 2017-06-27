import collections

import numpy as np
import tensorflow as tf

from utils import *



vgg19_npy_path = '/scratch/kvg245/vidsal_gan/vidsal_gan/output/SAL1_0noskip/model.npy'

def vgg_encoder(inputs):
    if vgg19_npy_path is not None:
	data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
    
    with tf.variable_scope('encoder_1'):
    	conv1_1 = conv_layer2(data_dict,inputs, 3, 64, "conv1_1")
    	conv1_2 = conv_layer2(data_dict,conv1_1, 64, 64, "conv1_2")
    	pool1 = max_pool(conv1_2, 'pool1')

    with tf.variable_scope('encoder_2'):
        conv2_1 = conv_layer2(data_dict,pool1, 64, 128, "conv2_1")
        conv2_2 = conv_layer2(data_dict,conv2_1, 128, 128, "conv2_2")
	conv2_2 = tf.contrib.slim.batch_norm(conv2_2)
        pool2 = max_pool(conv2_2, 'pool2')

    with tf.variable_scope('encoder_3'):
    	conv3_1 = conv_layer2(data_dict,pool2, 128, 256, "conv3_1")
    	conv3_2 = conv_layer2(data_dict,conv3_1, 256, 256, "conv3_2")
    	conv3_3 = conv_layer2(data_dict,conv3_2, 256, 256, "conv3_3")
    	conv3_4 = tf.contrib.slim.batch_norm(conv_layer2(data_dict,conv3_3, 256, 256, "conv3_4"))
    	pool3 = max_pool(conv3_4, 'pool3')

    with tf.variable_scope('encoder_4'):
    	conv4_1 = conv_layer2(data_dict,pool3, 256, 512, "conv4_1")
    	conv4_2 = conv_layer2(data_dict,conv4_1, 512, 512, "conv4_2")
    	conv4_3 = conv_layer2(data_dict,conv4_2, 512, 512, "conv4_3")
    	conv4_4 = tf.contrib.slim.batch_norm(conv_layer2(data_dict,conv4_3, 512, 512, "conv4_4"))
    	pool4 = max_pool(conv4_4, 'pool4')

    with tf.variable_scope('encoder_5'):

        conv5_1 = conv_layer2(data_dict,pool4, 512, 512, "conv5_1")
        conv5_2 = conv_layer2(data_dict,conv5_1, 512, 512, "conv5_2")
        conv5_3 = conv_layer2(data_dict,conv5_2, 512, 512, "conv5_3")
        conv5_4 = tf.contrib.slim.batch_norm(conv_layer2(data_dict,conv5_3, 512, 512, "conv5_4"))

    layers = [conv1_2, conv2_2, conv3_4, conv4_4, conv5_4]

    return layers
    
def vgg_decoder(encoded):
    if vgg19_npy_path is not None:
        data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
    layers = [encoded]
    with tf.variable_scope('decoder_1'):
	deconv1_1 = deconv_layer2(layers[-1],512,3,1,"deconv1_1",data_dict)
        deconv1_2 = deconv_layer2(deconv1_1,512,3,1,"deconv1_2",data_dict)
	deconv1_3 = deconv_layer2(deconv1_2,512,3,1,"deconv1_3",data_dict)
        deconv1_4 = deconv_layer2(deconv1_3,512,3,2,"deconv1_4",data_dict)
	out = tf.contrib.slim.batch_norm(deconv1_4)
	out = tf.nn.dropout(out,0.5)
	layers.append(out)
    
    with tf.variable_scope('decoder_2'):
        #deconv2_1 = deconv_layer(tf.concat([layers[-1],layers[3]],axis=3),256,3,1,"deconv2_1")
        deconv2_1 = deconv_layer2(layers[-1],256,3,1,"deconv2_1",data_dict)
	deconv2_2 = deconv_layer2(deconv2_1,256,3,1,"deconv2_2",data_dict)
        deconv2_3 = deconv_layer2(deconv2_2,256,3,1,"deconv2_3",data_dict)
        deconv2_4 = deconv_layer2(deconv2_3,256,3,2,"deconv2_4",data_dict)
 	out = tf.contrib.slim.batch_norm(deconv2_4)
	out = tf.nn.dropout(out,0.5)
	layers.append(out)

    with tf.variable_scope('decoder_3'):
        #deconv3_1 = deconv_layer(tf.concat([layers[-1],layers[2]],axis=3),128,3,1,"deconv3_1")
        deconv3_1 = deconv_layer2(layers[-1],128,3,1,"deconv3_1",data_dict)
	deconv3_2 = deconv_layer2(deconv3_1,128,3,1,"deconv3_2",data_dict)
        deconv3_3 = deconv_layer2(deconv3_2,128,3,1,"deconv3_3",data_dict)
        deconv3_4 = deconv_layer2(deconv3_3,128,3,2,"deconv3_4",data_dict)
        out = tf.contrib.slim.batch_norm(deconv3_4)
        out = tf.nn.dropout(out,0.5)
        layers.append(out)

    with tf.variable_scope('decoder_4'):
        #deconv4_1 = deconv_layer(tf.concat([layers[-1],layers[1]],axis=3),64,3,1,"deconv4_1")
        deconv4_1 = deconv_layer2(layers[-1],64,3,1,"deconv4_1",data_dict)
	deconv4_2 = deconv_layer2(deconv4_1,64,3,2,"deconv4_2",data_dict)
        out = tf.contrib.slim.batch_norm(deconv4_2)
        out = tf.nn.dropout(out,1.0)
        layers.append(out)

    with tf.variable_scope('decoder_5'):
        #deconv5_1 = deconv_layer(tf.concat([layers[-1],layers[0]],axis=3),32,3,1,"deconv5_1")
	deconv5_1 = deconv_layer2(layers[-1],64,3,1,"deconv5_1",data_dict)

        deconv5_2 = deconv_layer2(deconv5_1,64,3,1,"deconv5_2",data_dict)
        out = tf.nn.dropout(deconv5_2,1.0)
	output = conv112(out,data_dict)
        
        layers.append(output)
    return layers
    






 




def create_generator(generator_inputs, generator_outputs_channels):

    layers = vgg_encoder(generator_inputs)
    layers = vgg_decoder(layers)      
    return layers[-1]

