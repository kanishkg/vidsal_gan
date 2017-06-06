import os
from random import shuffle
import collections
import pickle

import numpy  as np
import tensorflow as tf
import h5py

from utils import *
from vgg_model import *

import subprocess as sp


Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_cross, gen_loss_L1,gen_nss, gen_grads_and_vars, train")

data_dir = '/scratch/kvg245/vidsal_gan/vidsal_gan/data/IRCCYN3D/'
output_dir = '/scratch/kvg245/vidsal_gan/vidsal_gan/output/vggtest/'
target_file = 'maps_data_saliency_224.h5'
input_file = 'maps_data_224.h5'
index_file = 'indices'

train = True
overfit = True
ckpt = False
num_past =1 
max_epoch = 300
seed = 4
num_frames = 4
progress_freq = 1
summary_freq = 100
save_freq = 1000
val_freq = 3000
vid_dict ={0: u'src01_hrc00_s3840x1080p25n400.avi', 1: u'src02_hrc00_s3840x1080p25n400.avi', 2: u'src03_hrc00_s3840x1080p25n400.avi', 3: u'src04_hrc00_s3840x1080p25n400.avi', 4: u'src05_hrc00_s3840x1080p25n400.avi', 5: u'src06_hrc00_s3840x1080p25n400.avi', 6: u'src07_hrc00_s3840x1080p25n400.avi', 7: u'src08_hrc00_s3840x1080p25n325.avi', 8: u'src09_hrc00_s3840x1080p25n400.avi', 9: u'src10_hrc00_s3840x1080p25n400.avi', 10: u'src11_hrc00_s3840x1080p25n400.avi', 11: u'src12_hrc00_s3840x1080p25n400.avi', 12: u'src13_hrc00_s3840x1080p25n400.avi', 13: u'src14_hrc00_s3840x1080p25n400.avi', 14: u'src15_hrc00_s3840x1080p25n400.avi', 15: u'src16_hrc00_s3840x1080p25n400.avi', 16: u'src17_hrc00_s3840x1080p25n400.avi', 17: u'src18_hrc00_s3840x1080p25n400.avi', 18: u'src19_hrc00_s3840x1080p25n400.avi', 19: u'src20_hrc00_s3840x1080p25n400.avi', 20: u'src21_hrc00_s3840x1080p25n400.avi', 21: u'src22_hrc00_s3840x1080p25n400.avi', 22: u'src23_hrc00_s3840x1080p25n400.avi', 23: u'src24_hrc00_s3840x1080p25n400.avi', 24: u'src25_hrc00_s3840x1080p25n400.avi', 25: u'src26_hrc00_s3840x1080p25n400.avi', 26: u'src27_hrc00_s3840x1080p25n400.avi', 27: u'src28_hrc00_s3840x1080p25n400.avi', 28: u'src29_hrc00_s3840x1080p25n400.avi', 29: u'src30_hrc00_s3840x1080p25n400.avi', 30: u'src31_hrc00_s3840x1080p25n400.avi', 31: u'src32_hrc00_s3840x1080p25n400.avi', 32: u'src33_hrc00_s3840x1080p25n400.avi', 33: u'src34_hrc00_s3840x1080p25n400.avi', 34: u'src35_hrc00_s3840x1080p25n400.avi', 35: u'src36_hrc00_s3840x1080p25n400.avi', 36: u'src37_hrc00_s3840x1080p25n400.avi', 37: u'src38_hrc00_s3840x1080p25n400.avi', 38: u'src39_hrc00_s3840x1080p25n400.avi', 39: u'src40_hrc00_s3840x1080p25n400.avi', 40: u'src41_hrc00_s3840x1080p25n400.avi', 41: u'src42_hrc00_s3840x1080p25n400.avi', 42: u'src43_hrc00_s3840x1080p25n400.avi', 43: u'src44_hrc00_s3840x1080p25n400.avi', 44: u'src45_hrc00_s3840x1080p25n400.avi', 45: u'src46_hrc00_s3840x1080p25n400.avi', 46: u'src47_hrc00_s3840x1080p25n400.avi'}

target_dict = {0: u'src01_hrc00_s1920x1080p25n400.avi', 1: u'src02_hrc00_s1920x1080p25n400.avi', 2: u'src03_hrc00_s1920x1080p25n400.avi', 3: u'src04_hrc00_s1920x1080p25n400.avi', 4: u'src05_hrc00_s1920x1080p25n400.avi', 5: u'src06_hrc00_s1920x1080p25n400.avi', 6: u'src07_hrc00_s1920x1080p25n400.avi', 7: u'src08_hrc00_s1920x1080p25n325.avi', 8: u'src09_hrc00_s1920x1080p25n400.avi', 9: u'src10_hrc00_s1920x1080p25n400.avi', 10: u'src11_hrc00_s1920x1080p25n400.avi', 11: u'src12_hrc00_s1920x1080p25n400.avi', 12: u'src13_hrc00_s1920x1080p25n400.avi', 13: u'src14_hrc00_s1920x1080p25n400.avi', 14: u'src15_hrc00_s1920x1080p25n400.avi', 15: u'src16_hrc00_s1920x1080p25n400.avi', 16: u'src17_hrc00_s1920x1080p25n400.avi', 17: u'src18_hrc00_s1920x1080p25n400.avi', 18: u'src19_hrc00_s1920x1080p25n400.avi', 19: u'src20_hrc00_s1920x1080p25n400.avi', 20: u'src21_hrc00_s1920x1080p25n400.avi', 21: u'src22_hrc00_s1920x1080p25n400.avi', 22: u'src23_hrc00_s1920x1080p25n400.avi', 23: u'src24_hrc00_s1920x1080p25n400.avi', 24: u'src25_hrc00_s1920x1080p25n400.avi', 25: u'src26_hrc00_s1920x1080p25n400.avi', 26: u'src27_hrc00_s1920x1080p25n400.avi', 27: u'src28_hrc00_s1920x1080p25n400.avi', 28: u'src29_hrc00_s1920x1080p25n400.avi', 29: u'src30_hrc00_s1920x1080p25n400.avi', 30: u'src31_hrc00_s1920x1080p25n400.avi', 31: u'src32_hrc00_s1920x1080p25n400.avi', 32: u'src33_hrc00_s1920x1080p25n400.avi', 33: u'src34_hrc00_s1920x1080p25n400.avi', 34: u'src35_hrc00_s1920x1080p25n400.avi', 35: u'src36_hrc00_s1920x1080p25n400.avi', 36: u'src37_hrc00_s1920x1080p25n400.avi', 37: u'src38_hrc00_s1920x1080p25n400.avi', 38: u'src39_hrc00_s1920x1080p25n400.avi', 39: u'src40_hrc00_s1920x1080p25n400.avi', 40: u'src41_hrc00_s1920x1080p25n400.avi', 41: u'src42_hrc00_s1920x1080p25n400.avi', 42: u'src43_hrc00_s1920x1080p25n400.avi', 43: u'src44_hrc00_s1920x1080p25n400.avi', 44: u'src45_hrc00_s1920x1080p25n400.avi', 45: u'src46_hrc00_s1920x1080p25n400.avi', 46: u'src47_hrc00_s1920x1080p25n400.avi'}

class batch_generator:
    """Provides batches of input and target files with training indices
       Also opens h5 datasets"""
    
    def __init__( self,batch_size = 8,istrain = True):
    	self.batch_size = batch_size
	self.istrain = istrain
	self.index_data,self.target_data,self.input_data = self.open_files()
	self.batch_len = len(self.index_data)
        self.current_epoch = None
        self.batch_index = None

    def open_files(self):
        """Opens h5 dataset files and loads index pickle"""
        with open(data_dir+index_file,'rb') as f:
	    index_raw= pickle.load(f)
            val_list = [6,13,20,34,41]
	    index_data = []
	    for a in index_raw:
		if self.istrain and a[0] not in val_list:
		    index_data.append(a)
		elif not self.istrain and a[0] in val_list:
		    index_data.append(a)
	    index_data = shuffled(index_data)
	print len(index_data)	
	input_list = []
	target_list = []

        target_data = h5py.File(data_dir+target_file,'r') 
        input_data = h5py.File(data_dir+input_file,'r')
	#for i in range(len(input_data.keys())):
	#    input_list.append(input_data[vid_dict[i]][:])
        #    target_list.append(target_data[vid_dict[i]][:])
	#    print i
	#with open(data_dir+'data','w') as f:
	#    data={'input':input_list,'target': target_list}
	#    pickle.dump(data,f)
        return index_data,target_data, input_data

    def create_batch(self, data_list):
	"""Creates and returns a batch of input and output data"""
	input_batch = []
	target_batch = []
#	progress  = ProgressBar(len(data_list),fmt=ProgressBar.FULL)
	
	for data in data_list:
    	    video =  self.input_data[vid_dict[data[0]]][:]
	    target = self.target_data[target_dict[data[0]]][:]
	    frames = np.asarray(video[data[1]])
	    maps = np.asarray(target[data[1]])
	    input_batch.append(frames)
	    target_batch.append(maps)
#	    progress.current+=1
 #           progress()
#	progress.done()
	target_batch = np.asarray(target_batch)/255.0
	input_batch = np.asarray(input_batch)
	VGG_MEAN = [103.939, 116.779, 123.68]
	input_batch[:,:,:,0] -= VGG_MEAN[0]
	input_batch[:,:,:,1] -= VGG_MEAN[1]
	input_batch[:,:,:,2] -= VGG_MEAN[2]
	#dummy = np.zeros((input_batch.shape[0],input_batch.shape[2],input_batch.shape[3],input_batch.shape[1]*input_batch.shape[4]))
	#dummy[:,:,:,:3] = input_batch[:,0,:,:,:]
	#dummy[:,:,:,3:6] = input_batch[:,1,:,:,:]
	#dummy[:,:,:,6:9] = input_batch[:,2,:,:,:]
	#dummy[:,:,:,9:] = input_batch[:,3,:,:,:]
	return {'input':np.reshape(input_batch,(input_batch.shape[0],input_batch.shape[1],input_batch.shape[2],input_batch.shape[3])),'target':np.reshape(target_batch,(target_batch.shape[0],target_batch.shape[1],target_batch.shape[2],1))}
		
    def get_batch_vec(self):
	"""Provides batch of data to process and keeps 
	track of current index and epoch"""
	
        if self.batch_index is None:
	    self.batch_index = 0
	    self.current_epoch = 0
	
        if self.batch_index < self.batch_len-self.batch_size-1:
	    batch_dict = self.create_batch(self.index_data[self.batch_index:self.batch_index + self.batch_size])
	    self.batch_index += self.batch_size
        else:
            self.current_epoch += 1
            self.batch_index = 0
            batch_dict = self.create_batch(self.index_data[self.batch_index:self.batch_index + self.batch_size])
            self.batch_index += self.batch_size


        return batch_dict
    
def main():
    
    if not os.path.exists(output_dir):
	os.makedirs(output_dir)
    batch_size = 4
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    	
    #examples = bg.get_batch_vec()
    #print examples['input'].shape , examples['target'].shape
    input = tf.placeholder(dtype = tf.float32,shape = (batch_size,224,224,num_past*3))
    target = tf.placeholder(dtype = tf.float32, shape = (batch_size,224,224,1))
    #examples = {'input':np.zeros((1,256,256,12)),'target':np.zeros((1,256,256,1))}
     
    
    model = create_model(input,target)
    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    tf.summary.scalar("generator_loss_cross",model.gen_loss_cross)
    tf.summary.scalar("generator_nss",model.gen_nss)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        print(var.op.name,grad)
	tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
    #if ckpt:                                                                
    #	new_saver = tf.train.import_meta_graph(output_dir+'model.ckpt.meta')
    saver = tf.train.Saver()
    restore_saver = tf.train.Saver()    
    
    sv = tf.train.Supervisor(logdir=output_dir, save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:
		
        if ckpt:
            print("loading model from checkpoint")
	    print(output_dir)
            checkpoint = tf.train.latest_checkpoint(output_dir)
	    restore_saver.restore(sess,checkpoint)
            
	if not train:
	    bg = batch_generator(batch_size,False)
	    
	    batch = bg.get_batch_vec()
	    tn =0 
	    k =0
	    tnss = 0
	    while bg.current_epoch == 0 :
		feed_dict = {input:batch['input'],target :batch['target']}
		predictions,_ = sess.run([model.outputs,model.gen_nss],feed_dict = feed_dict)
		
		for i in range(batch_size):
		    k+=1
		    p = predictions[i,:,:,:]*255.0
		    t = batch['target'][i,:,:,:]*255.0
		    n = batch['input'][i,:,:,0:2]
		    pss = p[:,:,0]
		    tss = t[:,:,0] 
		    kl = kld(pss,tss)
		    nss = nss_calc(tss,tss)
		    tnss+= nss
		    tn+=kl
		    #save_image(p,output_dir,str(bg.batch_index+i)+'p')
                    #save_image(t,output_dir,str(bg.batch_index+i)+'t')
                    #save_image(n,output_dir,str(bg.batch_index+i)+'i')
		    print(nss,tnss/k,kl,tn/k,bg.batch_index+i,bg.batch_len)
		batch = bg.get_batch_vec()	
		
	elif overfit:
	    bg = batch_generator(batch_size,False)
	    batch = bg.get_batch_vec()
	    feed_dict = {input:batch['input'],target :batch['target']} 
	    for i in range(max_epoch):
	        start = time.time()
                def should(freq):
                    return freq > 0 and ((bg.batch_index/batch_size) % freq == 0 )
                
                fetches = {
                   "train": model.train,
                   "global_step": sv.global_step,
                        }

                if should(summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1
		    fetches["gen_nss"] = model.gen_nss
		    fetches["gen_loss_cross"] = model.gen_loss_cross
		#a = sess.run(fetches,feed_dict=feed_dict)
                
		results = sess.run(fetches,feed_dict = feed_dict)

                print(results["discrim_loss"],results["gen_loss_GAN"],results['gen_loss_L1'],results['gen_loss_cross'],results["gen_nss"],i)
                if should(summary_freq):
                   print("recording summary")
                   sv.summary_writer.add_summary(results["summary"], i)
                if should(save_freq):
                   print("saving model")
                   saver.save(sess, output_dir+"model.ckpt")
	    predictions = sess.run(model.outputs,feed_dict = feed_dict)
            for i in range(batch_size):
                p = predictions[i,:,:,:]*255.0
                t = batch['target'][i,:,:,:]*255.0
                n = batch['input'][i,:,:,0:3]
                save_image(p,output_dir,str(bg.batch_index+i)+'p')
                save_image(t,output_dir,str(bg.batch_index+i)+'t')
                save_image(n,output_dir,str(bg.batch_index+i)+'i')
                print(i,bg.batch_len)



		
	elif train:
	    start = time.time()
	    bg = batch_generator(batch_size)
	    gv =0
            lv =0 
	    while bg.current_epoch<max_epoch:
	        c = bg.current_epoch
	        #progress = ProgressBar(bg.batch_len/bg.batch_size,fmt = ProgressBar.FULL)
	        while bg.current_epoch == c:
		    start = time.time()
		    def should(freq):
		        return freq > 0 and ((bg.batch_index/batch_size) % freq == 0 )
		    batch = bg.get_batch_vec()
		    feed_dict = {input:batch['input'],target :batch['target']}	
		    fetches = {
                        "train": model.train,
                        "global_step": sv.global_step,
			    }

		    if should(summary_freq):
		        fetches["summary"] = sv.summary_op

		    if should(progress_freq):
                        fetches["discrim_loss"] = model.discrim_loss
                        fetches["gen_loss_GAN"] = model.gen_loss_GAN
		        fetches["gen_loss_L1"] = model.gen_loss_L1
			fetches["gen_loss_cross"] = model.gen_loss_cross
			fetches["gen_nss"] = model.gen_nss	
		
		    results = sess.run(fetches,feed_dict = feed_dict)
		
		    print(results["discrim_loss"],results["gen_loss_GAN"],results['gen_loss_L1'],results['gen_loss_cross'],results['gen_nss'],gv,lv,bg.current_epoch,bg.batch_index,time.time()-start,(time.time()-start)*(bg.batch_len-bg.batch_index)/batch_size*(max_epoch-bg.current_epoch-2)*batch_size)
                    if should(summary_freq):
                        print("recording summary")
                        sv.summary_writer.add_summary(results["summary"], (bg.batch_index/bg.batch_size+bg.batch_len/batch_size*bg.current_epoch))
                    if should(save_freq):
                        print("saving model")
                        saver.save(sess, output_dir+"model.ckpt")
		    if should(val_freq):
                        bgv = batch_generator(batch_size,False)
                        batchv = bgv.get_batch_vec()
                        gv = 0
                        lv = 0
			cv = 0
                        while bgv.current_epoch == 0 :
                            feed_dictv = {input:batchv['input'],target :batchv['target']}
                            predictions,lgan,l1,cross = sess.run([model.outputs,model.gen_loss_GAN,model.gen_loss_L1,model.gen_loss_cross],feed_dict = feed_dictv)
                            gv+=lgan
                            lv+=l1
			    cv+=cross
			    batchv = bgv.get_batch_vec()
                        print("validation loss",gv,lv)


main()
