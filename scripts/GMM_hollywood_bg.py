import pickle
import numpy as np
from utils import *
import random
import h5py
import time

class batch_generator:

    def __init__( self,batch_size = 8,istrain = True, input_file = '/scratch/kvg245/vidsal_gan/vidsal_gan/data/Hollywood2/target2.h5',target_file = '/scratch/kvg245/vidsal_gan/vidsal_gan/data/Hollywood2/fix.npy',num_frames = 16,index_file = '/scratch/kvg245/vidsal_gan/vidsal_gan/data/Hollywood2/index2'):
        self.batch_size = batch_size
        self.istrain = istrain
        random.seed(4)
        self.val_list = list(random.sample(range(0,1705),int(1705*0.2)))
     	self.index_data,self.target_data,self.input_data = self.open_files(target_file,input_file,index_file)
        self.batch_len = len(self.index_data)
        self.current_epoch = None
        self.batch_index = None
     	self.num_frames = num_frames
	
        

    def open_files(self,target_file,input_file,index_file):
	
	input = h5py.File(input_file,'r') 
	target =  np.load(target_file)
	
	with open(index_file,'rb') as f:
	    index_data = pickle.load(f)
	if self.istrain:
	    index_data = shuffled([a for a in index_data if a[0] not in self.val_list])
	else:
	    index_data = shuffled([a for a in index_data if a[0] in self.val_list])
        
	print len(index_data)
        
	return index_data,target, input

    def create_batch(self, data_list):
        """Creates and returns a batch of input and output data"""
	
        input_batch = []
        target_batch = []
        for data in data_list:
	    #video = np.load('/scratch/kvg245/vidsal_gan/vidsal_gan/data/Hollywood2/AVIClips/'+str(data[0])+'i.npy')[data[1]-self.num_frames+1:data[1]+1,...]
	    
	    #target = np.load('/scratch/kvg245/vidsal_gan/vidsal_gan/data/Hollywood2/AVIClips/'+str(data[0])+'t.npy')[data[1],...]
            video =  self.input_data[str(data[0])][data[1]-self.num_frames+1:data[1]+1,...]
	    print len(self.target_data[data[0]])
	    print data[1]
            target_temp = np.asarray(self.target_data[data[0]][data[1]])
	    target = np.zeros((15,2),dtype=np.float32)
	    for i in range(target_temp.shape[0]):
		if i>=15:
		    break
		target[i,:] =  target_temp[i,:]
            VGG_MEAN = [103.939, 116.779, 123.68]
            video[:,:,:,0]-=VGG_MEAN[0]
            video[:,:,:,1]-=VGG_MEAN[1]
            video[:,:,:,2]-=VGG_MEAN[2]
            input_batch.append(video)
            target_batch.append(target)
        target_batch = np.asarray(target_batch)
        input_batch = np.asarray(input_batch)
        #dummy = np.zeros((input_batch.shape[0],input_batch.shape[2],input_batch.shape[3],input_batch.shape[1]*input_batch.shape[4]))
        #for i in range(self.num_frames-1):
        #    dummy[:,:,:,3*i:3*i+3] = input_batch[:,i,:,:,:]
 	
	return {'input':input_batch,'target':target_batch}

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

