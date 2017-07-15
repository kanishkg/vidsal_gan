import collections

from utils import *
from vid_vgg_model import *
import tensorflow as tf


Model = collections.namedtuple("Model", "encoding, outputs, gen_loss, gen_grads_and_vars, train")

l1_weight = 0.
cross_weight = 1.0

c3d_npy_path = 'weights/c3d.npy'

def temporal_encoder(inputs):

    if c3d_npy_path is not None:
        data_dict = np.load(c3d_npy_path, encoding='latin1').item()

    with tf.variable_scope('encoder3d_1'):
	
        conv1_1 = conv3d_layer2(data_dict,inputs, 3, 64 ,"conv1_1")
        pool1 = max_pool3d1(conv1_1, 'pool1')

    with tf.variable_scope('encoder3d_2'):
        conv2_1 = conv3d_layer2(data_dict,pool1, 64, 128, "conv2_1")
        norm = tf.contrib.slim.batch_norm(conv2_1)
        pool2 = max_pool3d(norm, 'pool2')

    with tf.variable_scope('encoder3d_3'):
        conv3_1 = conv3d_layer2(data_dict,pool2, 128, 256, "conv3_1")
        conv3_2 = conv3d_layer2(data_dict,conv3_1, 256, 256, "conv3_2")
        norm = tf.contrib.slim.batch_norm(conv3_2)
        pool3 = max_pool3d(norm, 'pool3')

    with tf.variable_scope('encoder3d_4'):
        conv4_1 = conv3d_layer2(data_dict,pool3, 256, 512, "conv4_1")
        conv4_2 = conv3d_layer2(data_dict,conv4_1, 512, 512, "conv4_2")
	norm = tf.contrib.slim.batch_norm(conv4_2)
	pool4 = max_pool3d(norm,'pool4')

    with tf.variable_scope('encoder3d_5'):
        conv5_1 = conv3d_layer2(data_dict,pool4, 512, 512, "conv5_1")
        conv5_2 = conv3d_layer2(data_dict,conv5_1, 512, 512, "conv5_2")
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
	net = max_pool(lrelu(encodings,0.2),'poolg')
	net = tf.contrib.layers.flatten(net,scope= 'flatteng')
	net = tf.contrib.slim.fully_connected(net,4096,scope = 'fc1g')
	net = tf.nn.dropout(net,0.5)
	net = tf.contrib.slim.fully_connected(net,4096,scope = 'fc2g')
	w1 = tf.get_variable('out_w',(4096,N*6),dtype = tf.float32)
	b1 = tf.get_variable('out_b',(N*6),dtype = tf.float32)
	params = tf.nn.xw_plus_b(net,w1,b1)
	#weights = tf.contrib.layers.fully_connected(net,N,activation_fn =tf.nn.softmax,biases_initializer = tf.ones_initializer(), scope = 'fc3wg')

	#params_v = tf.contrib.layers.fully_connected(net,N*2, biases_initializer=tf.ones_initializer(),scope ='fc3vg' )
    return params

def create_vid_model(inputs,targets,past,N=20):
    batch_size = 4
    with tf.variable_scope('generator'):
        with tf.device('/gpu:0'):
            current_frame = inputs[:,0,:,:,:]
    	    vgg = vgg_encoder (current_frame)
	with tf.device('/gpu:1'):
	    temporal = temporal_encoder (inputs)
	
    	encoded = combine_encodings(vgg[-1],temporal[-1],past)
    	params = gmm_params(encoded)
   
    with tf.name_scope("generator_loss"):
	NLL = tf.constant(0.0)
	#params_u= tf.unstack(params_u,axis =1)
	#params_v = tf.unstack(params_v,axis=1)
	#targets = tf.transpose(targets,[1,0,2])
	flat_target_data = tf.reshape(targets,[-1, 2])
        [x1_data, x2_data] = tf.split(axis=1, num_or_size_splits=2, value=flat_target_data)
        def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
         # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
	      norm1 = tf.subtract(x1, mu1)
	      norm2 = tf.subtract(x2, mu2)
	      s1s2 = tf.multiply(s1, s2)
	      z = tf.square(tf.div(norm1, s1))+tf.square(tf.div(norm2, s2))-2*tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)
	      negRho = 1-tf.square(rho)
	      result = tf.exp(tf.div(-z,2*negRho))
	      denom = 2*np.pi*tf.multiply(s1s2, tf.sqrt(negRho))
	      result = tf.div(result, denom)
	      return result

        def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, x1_data, x2_data):
	      result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
	      epsilon = 1e-20
	      result1 = tf.multiply(result0, z_pi)
	      result1 = tf.reduce_sum(result1, 1, keep_dims=True)
	      result1 = -tf.log(tf.maximum(result1, 1e-20)) # at the beginning, some errors are exactly zero.


	      return tf.reduce_sum(result1)

         # below is where we need to do MDN splitting of distribution params
        def get_mixture_coef(output):
	      # ie, eq 18 -> 23 of http://arxiv.org/abs/1308.0850
	      z = output
	      z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(axis=1, num_or_size_splits=6, value=z[:, :])

	      # process output z's into MDN paramters


	      # softmax all the pi's:
	      max_pi = tf.reduce_max(z_pi, 1, keep_dims=True)
	      z_pi = tf.subtract(z_pi, max_pi)
	      z_pi = tf.exp(z_pi)
	      normalize_pi = tf.reciprocal(tf.reduce_sum(z_pi, 1, keep_dims=True))
	      z_pi = tf.multiply(normalize_pi, z_pi)

	      # exponentiate the sigmas and also make corr between -1 and 1.
	      z_sigma1 = tf.exp(z_sigma1)
	      z_sigma2 = tf.exp(z_sigma2)
	      z_corr = tf.tanh(z_corr)

	      return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr]


        
	def get_tiled (out):
	    res = []
	    for o in out:
		res.append(tf.tile(o,[15,1]))
	    return res
	[o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr] = get_tiled(get_mixture_coef(params))	
	lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, x1_data, x2_data)
	
	#a = [] 
    	#for i in range(N):
	     
        #     pi,mu1,mu2,sig,ro = weights[:,i],params_u[i],params_u[N+i],params_v[i],params_v[N+i]
	#     sig+=0.000000001 
	#     ro+=0.000000001
	#     mu = tf.reshape([mu1,mu2],[batch_size,2])
	#     std = tf.reshape([sig,ro],[batch_size,2])
	#     s = 0
        #     dist = tf.contrib.distributions.MultivariateNormalDiag(loc = mu,scale_diag=std) 
	#     for j in range(15): 
	#         s+=pi*dist.prob(targets[j,:,:])
	#     NLL-= tf.reduce_sum(tf.log(s))
	gen_loss = lossfunc
	#gen_loss = tf.clip_by_value(gen_loss,0,1000)	
    with tf.name_scope("generator_train"):
        #gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        gen_optim = tf.train.AdamOptimizer()
        gen_grads_and_vars = gen_optim.compute_gradients(gen_loss)
        gen_train = gen_optim.apply_gradients( gen_grads_and_vars)

    #ema = tf.train.ExponentialMovingAverage(decay=0.99)
    #update_losses = ema.apply([gen_loss])
    #global_step = tf.contrib.framework.get_or_create_global_step()
    #incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        encoding = encoded,
        gen_loss=gen_loss,
        gen_grads_and_vars=gen_grads_and_vars,
        outputs = [params],
        train=tf.group(gen_train),
        )
