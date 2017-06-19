import os
import numpy as np
import h5py
import pickle
from utils import ProgressBar

lag = 4
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

input_dir = '/scratch/kvg245/vidsal_gan/vidsal_gan/data/Hollywood_2/AVIClips/input2.npy'
input  = np.load(input_dir)
index_list = []
for i in range(input.shape[0]):
    for j in range(input.shape[1]):
	if j>num_frames+lag:
	    index_list.append([i,j])

print "saving pickle"
with open('/scratch/kvg245/vidsal_gan/vidsal_gan/data/Hollywood_2/index','w') as f:
    pickle.dump(index_list,f)

