import os
from random import shuffle
import collections
import pickle

import numpy  as np
import tensorflow as tf
import h5py

from utils import *
from model import *

import subprocess as sp


Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")

data_dir = '/scratch/kvg245/vidsal_gan/vidsal_gan/data/savam/'
output_dir = '/scratch/kvg245/vidsal_gan/vidsal_gan/output/SAVAM_10_l1_100_1/'
target_file = 'gaussian_vizualizations/maps_data.h5'
input_file = 'video_data/input_data.h5'
index_file = 'video_data/indices'

train = True
overfit = False
ckpt = False
max_epoch = 30
seed = 4
num_frames = 4
progress_freq = 1
summary_freq = 100
save_freq = 1000
vid_dict = {0: 'v01_Hugo_2172_left.avi', 1: 'v02_Dolphin_131474_left.avi', 2: 'v03_StepUp_67443_left.avi', 3: 'v04_LIVE1_0_left.avi', 4: 'v05_LIVE2_0_left.avi', 5: 'v06_LIVE3_0_left.avi', 6: 'v07_Avatar_142222_left.avi', 7: 'v08_Dolphin_127156_left.avi', 8: 'v09_StepUpRevolution_119518_left.avi', 9: 'v10_VQEG01_0_left.avi', 10: 'v11_VQEG02_0_left.avi', 11: 'v12_VQEG03_0_left.avi', 12: 'v13_IntoTheDeep_36475_left.avi', 13: 'v14_Pirates_47241_left.avi', 14: 'v15_Sanctum_147749_left.avi', 15: 'v16_StepUp_17153_left.avi', 16: 'v17_SpiderMan_76686_left.avi', 17: 'v18_StepUp_76411_left.avi', 18: 'v19_Avatar_206134_left.avi', 19: 'v20_DriveAngry_2820_left.avi', 20: 'v21_Pirates_25246_left.avi', 21: 'v22_VQEG04_0_left.avi', 22: 'v23_VQEG05_0_left.avi', 23: 'v24_VQEG06_0_left.avi', 24: 'v25_Dolphin_17437_left.avi', 25: 'v26_Galapagos_14830_left.avi', 26: 'v27_UnderworldAwakening_69044_left.avi', 27: 'v28_VQEG10_0_left.avi', 28: 'v29_Avatar_46279_left.avi', 29: 'v30_Dolphin_79095_left.avi', 30: 'v31_DriveAngry_83142_left.avi', 31: 'v32_Dolphin_81162_left.avi', 32: 'v33_Hugo_87461_left.avi', 33: 'v34_VQEG07_0_left.avi', 34: 'v35_VQEG08_0_left.avi', 35: 'v36_VQEG09_0_left.avi', 36: 'v37_UnderworldAwakening_91276_left.avi', 37: 'v38_StepUp_82572_left.avi', 38: 'v39_Avatar_98125_left.avi', 39: 'v42_MSOffice_242_left.avi', 40: 'v43_Panasonic_373_left.avi', 41: 'video_data.h5', 42: 'video_data.h5'}


class batch_generator:
    """Provides batches of input and target files with training indices
       Also opens h5 datasets"""
    
    def __init__( self,batch_size = 8):
    	self.batch_size = batch_size
	
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
		if train and a[0] not in val_list:
		    index_data.append(a)
		elif not train and a[0] in val_list:
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
	    target = self.target_data[vid_dict[data[0]]][:]
	    frames = np.asarray([video[x] for x in data[1:]])
	    maps = np.asarray(target[data[1]])
	    input_batch.append(frames)
	    target_batch.append(maps)
#	    progress.current+=1
 #           progress()
#	progress.done()
	target_batch = np.asarray(target_batch)/255.0
	input_batch = np.asarray(input_batch)
	dummy = np.zeros((input_batch.shape[0],input_batch.shape[2],input_batch.shape[3],input_batch.shape[1]*input_batch.shape[4]))
	dummy[:,:,:,:3] = input_batch[:,0,:,:,:]
	dummy[:,:,:,3:6] = input_batch[:,1,:,:,:]
	dummy[:,:,:,6:9] = input_batch[:,2,:,:,:]
	dummy[:,:,:,9:] = input_batch[:,3,:,:,:]
	return {'input':dummy,'target':np.reshape(target_batch,(target_batch.shape[0],target_batch.shape[1],target_batch.shape[2],1))}
		
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
    bg = batch_generator(batch_size)
    	
    #examples = bg.get_batch_vec()
    #print examples['input'].shape , examples['target'].shape
    input = tf.placeholder(dtype = tf.float32,shape = (batch_size,256,256,12))
    target = tf.placeholder(dtype = tf.float32, shape = (batch_size,256,256,1))
    #examples = {'input':np.zeros((1,256,256,12)),'target':np.zeros((1,256,256,1))}
     
    
    model = create_model(input,target)
    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
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
            checkpoint = tf.train.latest_checkpoint(output_dir)
	    restore_saver.restore(sess,checkpoint)
            
	if not train:
	    bg = batch_generator(batch_size)
	    
	    batch = bg.get_batch_vec()
	    while bg.current_epoch == 0 :
		feed_dict = {input:batch['input'],target :batch['target']}
		predictions = sess.run(model.outputs,feed_dict = feed_dict)
		for i in range(batch_size):
		    p = predictions[i,:,:,:]*255.0
		    t = batch['target'][i,:,:,:]*255.0
		    print np.sort(p.reshape(256*256))
		    n = batch['input'][i,:,:,0:3]
		    #save_image(p,output_dir,str(bg.batch_index+i)+'p')
                   #save_image(t,output_dir,str(bg.batch_index+i)+'t')
                    #save_image(n,output_dir,str(bg.batch_index+i)+'i')
		    print(bg.batch_index+i,bg.batch_len)
		batch = bg.get_batch_vec()	
		
	elif overfit:
	    bg = batch_generator(batch_size)
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

                results = sess.run(fetches,feed_dict = feed_dict)

                print(results["discrim_loss"],results["gen_loss_GAN"],results['gen_loss_L1'],i)
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
		
		    results = sess.run(fetches,feed_dict = feed_dict)
		
		    print(results["discrim_loss"],results["gen_loss_GAN"],results['gen_loss_L1'],bg.current_epoch,bg.batch_index,time.time()-start,(time.time()-start)*(bg.batch_len-bg.batch_index)/batch_size*(max_epoch-bg.current_epoch-2)*batch_size)
                    if should(summary_freq):
                        print("recording summary")
                        sv.summary_writer.add_summary(results["summary"], (bg.batch_index/bg.batch_size+bg.batch_len/batch_size*bg.current_epoch))
                    if should(save_freq):
                        print("saving model")
                        saver.save(sess, output_dir+"model.ckpt")

main()
