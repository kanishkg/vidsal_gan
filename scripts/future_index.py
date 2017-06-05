import os
import numpy
import h5py
import pickle
from utils import ProgressBar

lag = 3
num_frames = 4
num_fut = 4
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
	 

	
        for f in range(data.shape[0]-num_fut):
	   
            
            if f<num_frames*lag-lag:
                pass
	    else:
		
                index_list.append([v])
		#print f,v
                for n in range (num_frames):
		    #print (v,f-n*lag)
                    index_list[len(index_list)-1].append(f-n*lag)
		for n in range(num_fut):
		    index_list[len(index_list)-1].append(f+n)
		
		
        v+=1
progress.done()
print "saving pickle"
with open(file_dir+'fut_indices','w') as ifile:
    pickle.dump(index_list,ifile)
