import collections
from utils import *
from eval import *
import numpy as np
import random
import tensorflow as tf
from GMM_model import *
from GMM_hollywood_bg import *
from utils import *
from tensorflow.python.client import timeline

Model = collections.namedtuple("Model", "encoding, outputs, gen_loss_cross, gen_loss_L1, gen_grads_and_vars, train")

mode = 'train'
ckpt = False

seed = 4
num_frames = 16
batch_size = 4 
max_epoch = 300

progress_freq = 1
save_freq = 3000
val_freq = 3
summary_freq = 400

output_dir = '/scratch/kvg245/vidsal_gan/vidsal_gan/output/gmmtest/'


def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    input = tf.placeholder(dtype = tf.float32,shape = (batch_size,num_frames,224,224,3))
    target = tf.placeholder(dtype = tf.float32, shape = (batch_size,15,2))
    past  = tf.placeholder(dtype = tf.float32,shape = (batch_size,14,14,512))
    print("Initializing the model")
    model = create_vid_model(input,target,past)
	

    print("Getting the Summaries in Order")    
    tf.summary.scalar("generator_loss", model.gen_loss)
    
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
	    batch = bg.get_batch_vec()
	    feed_dict = {input:batch['input'],target :batch['target']}
    	    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            sess.run(model.train, feed_dict = feed_dict,options=run_options, run_metadata=run_metadata)

            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(output_dir+'timeline.json', 'w') as f:
                f.write(ctf)	    
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
	    print(batch['target'].shape)
            feed_dict = {input:batch['input'],target :batch['target']}
            for i in range(max_epoch):
                start = time.time()
                def should(freq):
                    return freq > 0 and ((bg.batch_index/batch_size) % freq == 0 )

                fetches = {
                   "train": model.train,
                        }

                if should(progress_freq):
                    fetches["gen_loss"] = model.gen_loss

                results,_ = sess.run([model.gen_loss,model.train],feed_dict = feed_dict)

                print(results,i)
            
            predictions = sess.run(model.outputs,feed_dict = feed_dict)
            for i in range(batch_size):
                n = batch['input'][i,0,:,:,:]
                save_image(n,output_dir,str(bg.batch_index+i)+'i')

	else:
	    start = time.time()
            bg = batch_generator(batch_size)
	    
            bgv = batch_generator(batch_size,False)
            tl = 0
            bg.current_epoch = 0
            bg.batch_index = 72000
            while bg.current_epoch<max_epoch:
                c = bg.current_epoch
                l1_loss  = 0
                while bg.current_epoch == c:
                    start = time.time()
                    def should(freq):
                        return freq > 0 and ((bg.batch_index/batch_size) % freq == 0 )
                    batch = bg.get_batch_vec()
		    get_time = time.time()-start
                    feed_dict = {input:batch['input'],target :batch['target']}
                    fetches = {
                        "train": model.train,
			"encoding": model.encoding,
                            }

                    if should(summary_freq):
                        fetches["summary"] = sv.summary_op

                    if should(progress_freq):
                        fetches["gen_loss"] = model.gen_loss
		    here =time.time() 
		    results = sess.run(fetches,feed_dict = feed_dict)
		    compute_time = time.time()- here
		    print(get_time,compute_time)
		    print(results["encoding"])
                    l1_loss+= results['gen_loss']
                    print(l1_loss/bg.batch_index*batch_size,bg.current_epoch,bg.batch_index,time.time()-start,(time.time()-start)*(bg.batch_len-bg.batch_index)/batch_size+(max_epoch-bg.current_epoch-2)*bg.batch_len/batch_size)
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
		    j = 0
		    while bgv.current_epoch==0:
            	    	batchv = bgv.get_batch_vec()
                    	feed_dict = {input:batchv['input'],target :batchv['target']}
			l1,predictions = sess.run([model.gen_loss,model.outputs],feed_dict)
			tl+=l1
			j+=4.0
			for i in range(batch_size):
                	    n = batch['input'][i,0,:,:,:]
                	    save_image(n,output_dir,str(bgv.batch_index+i)+'i')
			    np.save(str(bgv.batch_index+i)+'t.npy',batchv['target'])
			    np.save(str(bgv.batch_index+i)+'p.npy',predictions)
                    print(j,tl/j)

main()
