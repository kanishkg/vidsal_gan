import collections
from utils import *
from eval import *
import numpy as np
import random
import tensorflow as tf
from video_model import *
from hollywood_bg import *
from utils import *


Model = collections.namedtuple("Model", "encoding, outputs, gen_loss_cross, gen_loss_L1, gen_grads_and_vars, train")

mode = 'train'
ckpt = False

seed = 4
num_frames = 16
batch_size = 4 
max_epoch = 300

progress_freq = 1
save_freq = 4000
val_freq = 3
summary_freq = 400

output_dir = '/scratch/kvg245/vidsal_gan/vidsal_gan/output/hrun2/'


def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    input = tf.placeholder(dtype = tf.float32,shape = (batch_size,num_frames,224,224,3))
    target = tf.placeholder(dtype = tf.float32, shape = (batch_size,224,224,1))
    past  = tf.placeholder(dtype = tf.float32,shape = (batch_size,14,14,512))
    print("Initializing the model")
    model = create_vid_model(input,target,past)
	

    print("Getting the Summaries in Order")    
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    tf.summary.scalar("generator_loss_cross",model.gen_loss_cross)
    
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.gen_grads_and_vars:

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

	if mode =='val':
            pass	    
        
	elif mode == "benchmark":
	    bg = batch_generator(batch_size)
	    
	    for i in range(50):
		start = time.time()
		batch = bg.get_batch_vec()
		read_time = time.time()
		print batch['input'].shape ,batch['target'].shape
		feed_dict = {input:batch['input'],target :batch['target']}
		_ = sess.run(model.outputs,feed_dict)
		fwd_pass = time.time()
		_ = sess.run(model.train,feed_dict)
		complete = time.time()
		print(i, read_time-start,fwd_pass-read_time,complete-fwd_pass)
	
        elif mode == 'overfit':
            bg = batch_generator(batch_size,False)
            batch = bg.get_batch_vec()
            feed_dict = {input:batch['input'],target :batch['target']}
            for i in range(max_epoch):
                start = time.time()
                def should(freq):
                    return freq > 0 and ((bg.batch_index/batch_size) % freq == 0 )

                fetches = {
                   "train": model.train,
                        }

                if should(progress_freq):
                    fetches["gen_loss_L1"] = model.gen_loss_L1
                    fetches["gen_loss_cross"] = model.gen_loss_cross

                results = sess.run(fetches,feed_dict = feed_dict)

                print(results['gen_loss_L1'],results['gen_loss_cross'],i)
            
            predictions = sess.run(model.outputs,feed_dict = feed_dict)
            for i in range(batch_size):
                p = predictions[i,:,:,:]
                t = batch['target'][i,:,:,:]
                n = batch['input'][i,0,:,:,:]
                save_image2(p[:,:,0],output_dir,str(bg.batch_index+i)+'p')
                save_image2(t[:,:,0],output_dir,str(bg.batch_index+i)+'t')
                save_image(n,output_dir,str(bg.batch_index+i)+'i')

	else:
	    start = time.time()
            bg = batch_generator(batch_size)
            bgv = batch_generator(batch_size,False)
            tl = 0
            tc = 0
            bg.current_epoch = 0
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
		    get_time = time.time()-start
                    feed_dict = {input:batch['input'],target :batch['target']}
                    fetches = {
                        "train": model.train,
                            }

                    if should(summary_freq):
                        fetches["summary"] = sv.summary_op

                    if should(progress_freq):
                        fetches["gen_loss_L1"] = model.gen_loss_L1
                        fetches["gen_loss_cross"] = model.gen_loss_cross
		    here =time.time() 
		    results = sess.run(fetches,feed_dict = feed_dict)
		    compute_time = time.time()- here
		    print(get_time,compute_time)
                    cross_loss+=results['gen_loss_cross']
                    l1_loss+= results['gen_loss_L1']
                    print(l1_loss/bg.batch_index*batch_size,cross_loss/bg.batch_index *batch_size,bg.current_epoch,bg.batch_index,time.time()-start,(time.time()-start)*(bg.batch_len-bg.batch_index)/batch_size+(max_epoch-bg.current_epoch-2)*bg.batch_len/batch_size)
                    if should(summary_freq):
                        print("recording summary")
                        sv.summary_writer.add_summary(results["summary"], (bg.batch_index/bg.batch_size+bg.batch_len/batch_size*bg.current_epoch))
                    if should(save_freq):
                        print("saving model")
                        saver.save(sess, output_dir+"model.ckpt")
		if (bg.current_epoch+1)%val_freq == 0:
	            bgv.current_epoch =0 
		    bgv.batch_index=0
		    tl = 0
		    tc = 0
		    j = 0
		    while bgv.current_epoch==0:
            	    	batchv = bgv.get_batch_vec()
                    	feed_dict = {input:batchv['input'],targetv :batch['target']}
			c,l1,predictions = sess.run([model.gen_loss_cross,gen_loss_L1,model.output],feed_dict)
		 	tc+=c
			tl+=l1
			j+=4.0
			for i in range(batch_size):
        	            p = predictions[i,:,:,:]
	                    t = batch['target'][i,:,:,:]
                	    n = batch['input'][i,0,:,:,:]
                	    save_image2(p[:,:,0],output_dir,str(bg.batch_index+i)+'p')
                	    save_image2(t[:,:,0],output_dir,str(bg.batch_index+i)+'t')
                	    save_image(n,output_dir,str(bg.batch_index+i)+'i')
                    print(j,tl/j,tc/j)

main()
