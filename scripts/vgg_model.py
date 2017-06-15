import collections

import numpy as np
import tensorflow as tf

from utils import *

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN,gen_loss_cross, gen_loss_L1,gen_nss, gen_grads_and_vars, train")

EPS = 1e-12
lr = 0.002
beta1 = 0.5
l1_weight = 0.0
gan_weight = 1.0
cross_weight = 20.0

vgg19_npy_path = '/scratch/kvg245/vidsal_gan/vidsal_gan/data/vgg19.npy'

def vgg_encoder(inputs):
    if vgg19_npy_path is not None:
	data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
    with tf.variable_scope('encoder_1'):
    	conv1_1 = conv_layer(data_dict,inputs, 3, 64, "conv1_1")
    	conv1_2 = conv_layer(data_dict,conv1_1, 64, 64, "conv1_2")
    	pool1 = max_pool(conv1_2, 'pool1')

    with tf.variable_scope('encoder_2'):
        conv2_1 = conv_layer(data_dict,pool1, 64, 128, "conv2_1")
        conv2_2 = conv_layer(data_dict,conv2_1, 128, 128, "conv2_2")
	conv2_2 = tf.contrib.slim.batch_norm(conv2_2)
        pool2 = max_pool(conv2_2, 'pool2')

    with tf.variable_scope('encoder_3'):
    	conv3_1 = conv_layer(data_dict,pool2, 128, 256, "conv3_1")
    	conv3_2 = conv_layer(data_dict,conv3_1, 256, 256, "conv3_2")
    	conv3_3 = conv_layer(data_dict,conv3_2, 256, 256, "conv3_3")
    	conv3_4 = tf.contrib.slim.batch_norm(conv_layer(data_dict,conv3_3, 256, 256, "conv3_4"))
    	pool3 = max_pool(conv3_4, 'pool3')

    with tf.variable_scope('encoder_4'):
    	conv4_1 = conv_layer(data_dict,pool3, 256, 512, "conv4_1")
    	conv4_2 = conv_layer(data_dict,conv4_1, 512, 512, "conv4_2")
    	conv4_3 = conv_layer(data_dict,conv4_2, 512, 512, "conv4_3")
    	conv4_4 = tf.contrib.slim.batch_norm(conv_layer(data_dict,conv4_3, 512, 512, "conv4_4"))
    	pool4 = max_pool(conv4_4, 'pool4')

    with tf.variable_scope('encoder_5'):

        conv5_1 = conv_layer(data_dict,pool4, 512, 512, "conv5_1")
        conv5_2 = conv_layer(data_dict,conv5_1, 512, 512, "conv5_2")
        conv5_3 = conv_layer(data_dict,conv5_2, 512, 512, "conv5_3")
        conv5_4 = tf.contrib.slim.batch_norm(conv_layer(data_dict,conv5_3, 512, 512, "conv5_4"))

    layers = [conv1_2, conv2_2, conv3_4, conv4_4, conv5_4]

    return layers
    
def vgg_decoder(layers):
    with tf.variable_scope('decoder_1'):
	deconv1_1 = deconv_layer(layers[-1],512,3,1,"deconv1_1")
        deconv1_2 = deconv_layer(deconv1_1,512,3,1,"deconv1_2")
	deconv1_3 = deconv_layer(deconv1_2,512,3,1,"deconv1_3")
        deconv1_4 = deconv_layer(deconv1_3,512,3,2,"deconv1_4")
	out = tf.contrib.slim.batch_norm(deconv1_4)
	out = tf.nn.dropout(out,0.5)
	layers.append(out)
    
    with tf.variable_scope('decoder_2'):
        #deconv2_1 = deconv_layer(tf.concat([layers[-1],layers[3]],axis=3),256,3,1,"deconv2_1")
        deconv2_1 = deconv_layer(layers[-1],256,3,1,"deconv2_1")
	deconv2_2 = deconv_layer(deconv2_1,256,3,1,"deconv2_2")
        deconv2_3 = deconv_layer(deconv2_2,256,3,1,"deconv2_3")
        deconv2_4 = deconv_layer(deconv2_3,256,3,2,"deconv2_4")
 	out = tf.contrib.slim.batch_norm(deconv2_4)
	out = tf.nn.dropout(out,0.5)
	layers.append(out)

    with tf.variable_scope('decoder_3'):
        #deconv3_1 = deconv_layer(tf.concat([layers[-1],layers[2]],axis=3),128,3,1,"deconv3_1")
        deconv3_1 = deconv_layer(layers[-1],128,3,1,"deconv3_1")
	deconv3_2 = deconv_layer(deconv3_1,128,3,1,"deconv3_2")
        deconv3_3 = deconv_layer(deconv3_2,128,3,1,"deconv3_3")
        deconv3_4 = deconv_layer(deconv3_3,128,3,2,"deconv3_4")
        out = tf.contrib.slim.batch_norm(deconv3_4)
        out = tf.nn.dropout(out,0.5)
        layers.append(out)

    with tf.variable_scope('decoder_4'):
        #deconv4_1 = deconv_layer(tf.concat([layers[-1],layers[1]],axis=3),64,3,1,"deconv4_1")
        deconv4_1 = deconv_layer(layers[-1],64,3,1,"deconv4_1")
	deconv4_2 = deconv_layer(deconv4_1,64,3,2,"deconv4_2")
        out = tf.contrib.slim.batch_norm(deconv4_2)
        out = tf.nn.dropout(out,1.0)
        layers.append(out)

    with tf.variable_scope('decoder_5'):
        #deconv5_1 = deconv_layer(tf.concat([layers[-1],layers[0]],axis=3),32,3,1,"deconv5_1")
	deconv5_1 = deconv_layer(layers[-1],64,3,1,"deconv5_1")

        deconv5_2 = deconv_layer(deconv5_1,64,3,1,"deconv5_2")
        out = tf.nn.dropout(deconv5_2,1.0)
	output = conv11(out)
        
        layers.append(output)
    return layers
    






 




def create_generator(generator_inputs, generator_outputs_channels):

    layers = vgg_encoder(generator_inputs)
    layers = vgg_decoder(layers)      
    return layers[-1]



def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 2
        layers = []
	ndf = 32 # numer of discriminator filters

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = conv(input, ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = tf.contrib.slim.batch_norm(convolved)
                rectified = tf.nn.relu(normalized)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)
        return layers[-1]

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, tf.sigmoid(outputs))

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
	gen_loss_L1 = tf.reduce_mean(tf.abs(targets - tf.sigmoid(outputs)))
	#gen_loss_cross = -tf.reduce_mean(targets*tf.log(outputs+EPS)+(1-targets)*tf.log(1-outputs+EPS))
	gen_loss_cross = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = targets,logits = outputs))
	maps = tf.map_fn(lambda img: tf.image.per_image_standardization(img), targets)
	gen_nss = tf.reduce_mean(maps*tf.sign(targets))
        gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight + gen_loss_cross * cross_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(lr*0.1, beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr,beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss,gen_loss_GAN,gen_loss_L1,gen_loss_cross])
    #update_losses = ema.apply([gen_loss_L1])
    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=discrim_loss,
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=gen_loss_GAN,
	gen_loss_cross = gen_loss_cross,
        gen_loss_L1=gen_loss_L1,
	gen_nss = gen_nss,
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=tf.sigmoid(outputs),
        train=tf.group(update_losses,incr_global_step, gen_train),
    )

