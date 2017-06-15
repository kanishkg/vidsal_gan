class batch_generator:

    def __init__( self,batch_size = 8,istrain = True, target_file = '/scratch/kvg245/vidsal_gan/vidsal_gan/data/Hollywood_2/AVIClips/input.npy',input_file = '/scratch/kvg245/vidsal_gan/vidsal_gan/data/Hollywood_2/AVIClips/target.npy',num_frames = 16):
        self.batch_size = batch_size
        self.istrain = istrain
        self.index_data,self.target_data,self.input_data = self.open_files(target_file,input_file)
        self.batch_len = len(self.index_data)
        self.current_epoch = None
        self.batch_index = None
     	self.num_frames = num_frames

    def open_files(self,target_file,input_file):
        with open(input_file,'r') as f:
	    input = np.asarray(np.load(f))
 	with open(target_file,'r') as f:
	    target = np.asarray(np.load(f))

        return index_data,target, input

    def create_batch(self, data_list):
        """Creates and returns a batch of input and output data"""
        input_batch = []
        target_batch = []

        for data in data_list:
            video =  self.input_data[vid_dict[data[0]]][:]
            target = self.target_data[target_dict[data[0]]][:]
            frames = np.asarray(video[data[1]])
            maps = np.asarray(target[data[1]])
            input_batch.append(frames)
            target_batch.append(maps)
        target_batch = np.asarray(target_batch)/255.0
        input_batch = np.asarray(input_batch)
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

