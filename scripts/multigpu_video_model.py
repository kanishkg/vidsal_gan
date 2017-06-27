import collections

from utils import *
from vid_vgg_model import *
import tensorflow as tf


Model = collections.namedtuple("Model", "encoding, outputs, gen_loss_cross, gen_loss_L1, gen_grads_and_vars, train")

l1_weight = 0.
cross_weight = 1.0

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.


    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis = 0, values = grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads



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

def inference_loss(inputs,targets,past):
    with tf.variable_scope('generator'):
        current_frame = inputs[:,0,:,:,:]
        vgg = vgg_encoder (current_frame)
        temporal = temporal_encoder (inputs)

        encoded = combine_encodings(vgg[-1],temporal[-1],past)
        outputs = vgg_decoder(encoded)[-1]

    with tf.name_scope("generator_loss"):
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - tf.sigmoid(outputs)))
        gen_loss_cross = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = targets,logits = outputs))
        gen_loss = gen_loss_L1 * l1_weight + gen_loss_cross * cross_weight

    with tf.name_scope("generator_train"):
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        gen_optim = tf.train.AdamOptimizer()
        gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
   
 
    return Model(
        encoding = encoded,
        gen_loss_cross = gen_loss_cross,
        gen_loss_L1=gen_loss_L1,
        gen_grads_and_vars= gen_grads_and_vars,
        outputs = tf.sigmoid(outputs),
        train=0,
        )


def create_vid_model(inputs,targets,past):
    bs = 4
    num_gpus = 1 
    towers = []
    tower_grads = []
    for i in xrange(num_gpus):
	with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    # grab this portion of the input
                    inp = inputs[i*bs:(i+1)*bs,...]
		    tn = targets[i*bs:(i+1)*bs,...]
		    pn = past[i*bs:(i+1)*bs,...]

		    towers.append( inference_loss(inp,tn,pn))
        	    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
		    gen_optim = tf.train.AdamOptimizer()
        	    gen_grads_and_vars = gen_optim.compute_gradients(towers[-1].gen_loss_cross, var_list=gen_tvars)
		    tower_grads.append(gen_grads_and_vars)
	
	avg_grads = average_gradients(tower_grads)
        #avg_grads = average_gradients([t.gen_grads_and_vars for t in towers] )
	gen_train = gen_optim.apply_gradients(avg_grads)
	encoding=[]
	output = []
	gen_loss_L1 = 0
	gen_loss_cross = 0
	tf.get_variable_scope().reuse_variables()
	for tower in towers:
	    gen_loss_L1+= tower.gen_loss_L1
	    gen_loss_cross+=tower.gen_loss_cross
	    output.append(tower.outputs)
	    encoding.append(tower.encoding)
	gen_loss_L1/=num_gpus
	gen_loss_cross/=num_gpus
	outputs = tf.concat(output,axis =0)
	encoded = tf.concat(encoding,axis=0)
		    
    return Model(
        encoding = encoded,
        gen_loss_cross = gen_loss_cross,
        gen_loss_L1=gen_loss_L1,
        gen_grads_and_vars=avg_grads,
        outputs = tf.sigmoid(outputs),
        train=tf.group(gen_train),
        )
