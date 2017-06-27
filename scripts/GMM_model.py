import collections

from utils import *
from vid_vgg_model import *
import tensorflow as tf


Model = collections.namedtuple("Model", "encoding, outputs, gen_loss_cross, gen_loss_L1, gen_grads_and_vars, train")

l1_weight = 0.
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
        conv4_1 = conv3d_layer(pool3, 256, 512, "conv4_1")
        conv4_2 = conv3d_layer(conv4_1, 512, 512, "conv4_2")
	norm = tf.contrib.slim.batch_norm(conv4_2)
	pool4 = max_pool3d(norm,'pool4')

    with tf.variable_scope('encoder3d_5'):
        conv5_1 = conv3d_layer(pool4, 512, 512, "conv5_1")
        conv5_2 = conv3d_layer(conv5_1, 512, 512, "conv5_2")
        norm = tf.contrib.slim.batch_norm(conv5_2)
    layers = [pool1,pool2,pool3,pool4,norm]
    return layers
    
def combine_encodings(vgg, temporal, past):
    with tf.variable_scope('concatinate'):
#	shape = tf.shape(vgg)
	combined = tf.concat([vgg,temporal[:,0,:,:,:],temporal[:,1,:,:,:]],axis = 3)
#	combined = tf.concat([tf.reshape(vgg,[shape[0],1,shape[2],shape[2],shape[3]]),temporal],axis =1)
    	
    with tf.variable_scope('weight_map'):
	conv6_11 = conv113(combined,512)
#	maps = tf.unstack(combined,axis = 1)
#	maps = tf.concat(maps,axis = 3)    
#	conv6_1 = conv_layer({},maps,512*3,512,"conv6_1")
#        fc_input = tf.contrib.layers.flatten(conv6_1) 
#        fc6_1 = tf.contrib.slim.fully_connected(fc_input,512*3,scope = "fc6_1")
#	out6 = 512 * tf.nn.softmax(fc6_1)

#    def feat_multiply(a,b):
#	return tf.stack([a[i,...]*b[i]for i in range(4)])
#    maps = []
#    for i in range(512):
	
#	maps.append(feat_multiply(vgg[:,:,:,i],out6[:,i])+feat_multiply(temporal[:,0,:,:,i],out6[:,i+512])+feat_multiply(temporal[:,1,:,:,i],out6[:,i+1024]))

#    vgg =tf.stack(maps,axis =3)
    
    return conv6_11
#    return vgg

def gmm_params(encodings, N = 20):
    with tf.variable_scope('gmm_pred'):
	net = max_pool(encodings,'poolg')
	net = tf.contribs.slim.ops.flatten(net,scope= 'flatteng')
	net = tf.contribs.slim.ops.fc(net,4096,scope = 'fc1g')
	net = tf.nn.dropout(net,0.5)
	net = tf.contribs.slim.ops.fc(net,4096.scope = 'fc2g')
	weights = tf.contribs.slim.ops.fc(net,N,activation = softmax, scope = 'fc3wg')	
	params = tf.contribs.slim.ops.fc(net, N*3,activation = None, scope = 'fc3pg')
    return weights,params

def create_vid_model(inputs,targets,past):
    with tf.variable_scope('generator'):
        current_frame = inputs[:,0,:,:,:]
    	vgg = vgg_encoder (current_frame)
	temporal = temporal_encoder (inputs)
	
    	encoded = combine_encodings(vgg[-1],temporal[-1],past)
    	weights, params = gmm_params(encoded)
   
    with tf.name_scope("generator_loss"):
         

    with tf.name_scope("generator_train"):
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
