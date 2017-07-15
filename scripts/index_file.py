import os
import numpy as np
import h5py
import pickle
from utils import ProgressBar

lag = 1
num_frames = 16
"""
file_dir = '../../../../scratch/kvg245/vidsal_gan/vidsal_gan/data/IRCCYN3D/'
data_list = []
print "reading file"

index_list = []
v = 0
with h5py.File(file_dir+'maps_data.h5','r') as hf:
    progress = ProgressBar(len(hf.keys()),fmt=ProgressBar.FULL)
    for file in sorted(hf.keys()):
    	progress.current+=1
	progress()
	data = hf[file][:]
	 

	
        for f in range(data.shape[0]):
	   
            
            if f<num_frames*lag-lag:
                pass
	    else:
		
                index_list.append([v])
		#print f,v
                for n in range (num_frames):
		    #print (v,f-n*lag)
                    index_list[len(index_list)-1].append(f-n*lag)
		
		
        v+=1
progress.done()
print "saving pickle"
with open(file_dir+'indices','w') as ifile:
    pickle.dump(index_list,ifile)
"""

input_dir = '/scratch/kvg245/vidsal_gan/vidsal_gan/data/Hollywood2/input2.h5'

index_list = []

with h5py.File(input_dir,'r') as hf:
    for i in sorted(hf.keys()):
	data = hf[i][:]
	print(data.shape)
    	for j in range(data.shape[0]):
	    if j>num_frames+lag:
	        index_list.append([int(i),j])
	print i

print "saving pickle"
print len(index_list)
with open('/scratch/kvg245/vidsal_gan/vidsal_gan/data/Hollywood2/index2','w') as f:
    pickle.dump(index_list,f)

