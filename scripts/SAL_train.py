import os
from random import shuffle
import collections
import pickle

import numpy  as np
import tensorflow as tf
import matplotlib.pyplot as plt
from salicon.salicon import SALICON
from utils import *
from vgg_model import *
from eval import *
import subprocess as sp
import cv2

Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_cross, gen_loss_L1,gen_nss, gen_grads_and_vars, train")

data_dir = '/scratch/kvg245/vidsal_gan/vidsal_gan/data/salicon-api/'
output_dir = '/scratch/kvg245/vidsal_gan/vidsal_gan/output/2SAL20_1/'
train_target_file = 'train2.npy'
train_input_file = 'train_input.npy'
val_target_file = 'val2.npy'
val_input_file = 'val_input.npy'


train = False
overfit = False
ckpt = True
num_past =1
max_epoch =300
seed =4
num_frames = 4
progress_freq = 1
summary_freq = 200
save_freq = 2499
val_freq = 5


class batch_generator:
    
    def __init__( self,batch_size = 8,istrain = True):
    	self.batch_size = batch_size
	self.istrain = istrain
	self.index_data,self.target_data,self.input_data = self.open_files()
	
	self.batch_len = len(self.index_data)    
        self.current_epoch = None
        self.batch_index = None

    def open_files(self):

	if self.istrain:
       	    index_list = shuffled(list(range(10000)))
	    target_data = np.load(data_dir+train_target_file)
	    input_data = np.load(data_dir+train_input_file)
	else:
	    index_list  = list(range(5000))
	    print("loading val tar")
	    target_data = np.load(data_dir+val_target_file)
	    print("loading val in")
            input_data = np.load(data_dir+val_input_file)

	return index_list,target_data,input_data

    def create_batch(self, data_list):
	"""Creates and returns a batch of input and output data"""
	input_batch = []
	target_batch = []
	
	for index in data_list:
	    
	    input_batch.append(self.input_data[index,:,:,:])
	    target_batch.append(self.target_data[index,:,:])


	target_batch = np.asarray(target_batch)
	input_batch = np.asarray(input_batch)

	VGG_MEAN = [103.939, 116.779, 123.68]
	input_batch[:,:,:,0] -= VGG_MEAN[0]
	input_batch[:,:,:,1] -= VGG_MEAN[1]
	input_batch[:,:,:,2] -= VGG_MEAN[2]
	return {'input': input_batch,'target':np.reshape(target_batch,(target_batch.shape[0],target_batch.shape[1],target_batch.shape[2],1))}
		
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
    
    	
    input = tf.placeholder(dtype = tf.float32,shape = (batch_size,224,224,3))
    target = tf.placeholder(dtype = tf.float32, shape = (batch_size,224,224,1))
     
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
	tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
    
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
	    ann_file = '/scratch/kvg245/vidsal_gan/vidsal_gan/data/salicon-api/annotations/fixations_val2014.json'
	    salicon =SALICON(ann_file)
	    ImgIds = sorted(salicon.getImgIds())
	    gt_fix = []
	    dims = []
	    for img in ImgIds:
		image = salicon.loadImgs(img)[0]
		dims.append([image['height'],image['width']])
		annid = salicon.getAnnIds(imgIds = img)
		anns = salicon.loadAnns(annid)
		fixations = [ann['fixations'] for ann in anns] 
		merged_fixations = [item for sublist in fixations for item in sublist] 
		gt_fix.append(merged_fixations)
 	    
	    k=0
	    bg = batch_generator(batch_size,False)
	    batch = bg.get_batch_vec()
	    tkl = 0 
	    tsim = 0
 	    tcc = 0
	    tnss = 0
	    taucb = 0
	    while bg.current_epoch == 0 :
		feed_dict = {input:batch['input'],target :batch['target']}
		predictions,_ = sess.run([model.outputs,model.gen_nss],feed_dict = feed_dict)
	    	save_image(batch['target'][0,...],output_dir,'00000')	
		for i in range(batch_size):
		    
		    p = predictions[i,...]
		    t = batch['target'][i,...]
		    n = batch['input'][i,:,:,0:3]
		    pss = p[:,:,0]
		    tss = t[:,:,0] 
		    kl = kld(pss,tss)
		    simv = sim(pss,tss)
                    ccv = cc(pss,tss) 
		    
		    tkl+=kl
		    tcc+=ccv
		    tsim+= simv

		    gt = np.asarray(gt_fix[k])
		    nssv = nss2(pss,gt,dims[k][0],dims[k][1]) 
		    auc= aucb(pss,gt,dims[k][0],dims[k][1])
		    taucb+=auc
		    tnss+=nssv
		    image = salicon.loadImgs(ImgIds[k])[0]
		    I = cv2.imread('/scratch/kvg245/vidsal_gan/vidsal_gan/data/salicon-api/images/'+image['file_name'])
		    #save_image(predictions[i,...]*255.0,output_dir,str(bg.batch_index+i)+'p')
		    #save_image(batch['target'][i,...]*255.0,output_dir,str(bg.batch_index+i)+'t')
		    #save_image(p,output_dir,str(bg.batch_index+i)+'p')
                    #save_image(t,output_dir,str(bg.batch_index+i)+'t')
                    #save_image(n,output_dir,str(bg.batch_index+i)+'i')
		    #save_image(n,'../',str(bg.batch_index+i)+'i')
		    #save_image(I,'../',str(bg.batch_index+i)+'o')
		    ex = bg.batch_index+i+1-4
		    k+=1
		    print(auc,taucb/ex,nssv,tnss/ex,simv,tsim/ex,ccv,tcc/ex,bg.batch_index+i,bg.batch_len)

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
                p = predictions[i,:,:,:]
                t = batch['target'][i,:,:,:]
                n = batch['input'][i,:,:,0:3]
                save_image2(p[:,:,0],output_dir,str(bg.batch_index+i)+'p')
                save_image2(t[:,:,0],output_dir,str(bg.batch_index+i)+'t')
                save_image(n,output_dir,str(bg.batch_index+i)+'i')
                print(i,bg.batch_len)



		
	elif train:
	    start = time.time()
	    bg = batch_generator(batch_size)
	    bgv = batch_generator(batch_size,False)
	    gv =0
            lv = 0
	    bg.current_epoch = 46
	    bg.batch_index = 0
	    while bg.current_epoch<max_epoch:
	        c = bg.current_epoch
	        cross_loss = 0
		l1_loss  = 0
		gan_loss = 0 
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
		    cross_loss+=results['gen_loss_cross']
		    l1_loss+= results['gen_loss_L1']
		    gan_loss+= results['gen_loss_GAN']
		    print(results["discrim_loss"],gan_loss/bg.batch_index*batch_size,l1_loss/bg.batch_index*batch_size,cross_loss/bg.batch_index *batch_size,results['gen_nss'],gv,lv,bg.current_epoch,bg.batch_index,time.time()-start,(time.time()-start)*(bg.batch_len-bg.batch_index)/batch_size+(max_epoch-bg.current_epoch-2)*bg.batch_len/batch_size)
                    if should(summary_freq):
                        print("recording summary")
                        sv.summary_writer.add_summary(results["summary"], (bg.batch_index/bg.batch_size+bg.batch_len/batch_size*bg.current_epoch))
                    if should(save_freq):
                        print("saving model")
                        saver.save(sess, output_dir+"model.ckpt")
	        if (bg.current_epoch+1)%val_freq==0:
		    print("validating")
                    bgv.current_epoch = 0
		    bgv.batch_index = 0
                    batchv = bgv.get_batch_vec()
		    
                    lv = 0
		    cv = 0
                    while bgv.current_epoch == 0 :
                        feed_dictv = {input:batchv['input'],target :batchv['target']}
                        predictions,l1,cross = sess.run([model.outputs,model.gen_loss_L1,model.gen_loss_cross],feed_dict = feed_dictv)
                        for i in range(batch_size):
                       	    p = predictions[i,:,:,:]
                	    t = batchv['target'][i,:,:,:]
                	    n = batchv['input'][i,:,:,0:3]
                	    save_image2(p[:,:,0],output_dir,str(bgv.batch_index+i)+'p')
                	    save_image2(t[:,:,0],output_dir,str(bgv.batch_index+i)+'t')
                	    save_image(n,output_dir,str(bgv.batch_index+i)+'i')
                    	lv+=l1
		     	cv+=cross
		    	batchv = bgv.get_batch_vec()
             	    print("validation loss",lv,cv)
main()
