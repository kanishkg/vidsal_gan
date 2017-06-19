import collections

from utils import *
from vim_vgg_model import *
import tensorflow as tf


Model = collections.namedtuple("Model", "encoding, outputs, gen_loss_GAN, gen_loss_cross, gen_loss_L1,gen_nss, gen_grads_and_vars, train")

l1_weight = 0.0
cross_weight = 1.0

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
    layers = [pool1,pool2,pool3,pool4,norm]
    return layers
    
def combine_encodings(vgg, temporal, past):
    with tf.variable_scope('concatinate'):
	combined = tf.concat(tf.concat(vgg,temporal,axis=4),past,axis = 4)

    with tf.variable_scope('weight_generator'):
	conv6_1 = conv_layer({},combined,512*4,512,"conv6_1")
        fc_input = tf.contrib.layers.flatten(conv6_1) 
        fc6_1 = tf.contrib.slim.fully_connected(fc_input,512*4,scope = "fc6_1")
	out6 = 512 * tf.nn.softmax(fc7)
    
    for i in range(512):
	vgg[:,:,:,i] = vgg[:,:,:,i]*out6[:i]+temporal[:,:,:,i]*out6[:,i+512*1]+temporal[:,:,:,i+1]*out6[:,:,:,i+512*2]+past[:,:,:,i]*out[:,i*512*3]
    
    return vgg

def create_vid_model(inputs,targets,past):
    with tf.variable_scope('generator'):
        current_frame = inputs[:,:,:,:3]
    	vgg = vgg_encoder (current_frame)
    	temporal = temporal_encoder (inputs)
    	encoded = combine_encodings(vgg[-1],temporal[-1],past)
    	outputs = vgg_decoder(encoded)
   
    with tf.name_scope("generator_loss"):
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - tf.sigmoid(outputs)))
        gen_loss_cross = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = targets,logits = outputs))
        gen_loss = gen_loss_L1 * l1_weight + gen_loss_cross * cross_weight

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer()
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([gen_loss_L1,gen_loss_cross])
    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        encoding = encoded,
        gen_loss_cross = gen_loss_cross,
        gen_loss_L1=gen_loss_L1,
        gen_grads_and_vars=gen_grads_and_vars,
        outputs = tf.sigmoid(outputs),
        train=tf.group(update_losses,incr_global_step, gen_train),
        )
