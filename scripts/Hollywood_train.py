import collections
from utils import *
from eval import *
import numpy as np
import random
import tensorflow as tf



def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    batch_size = 4
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    input = tf.placeholder(dtype = tf.float32,shape = (batch_size,256,256,num_frames*3))
    target = tf.placeholder(dtype = tf.float32, shape = (batch_size,256,256,1))


    model = create_model(input,target)

    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    tf.summary.scalar("generator_loss_cross",model.gen_loss_cross)
    
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

	if mode =='val':
            pass	    
	elif mode == 'overfit':
            pass
	else:
	    start = time.time()
            bg = batch_generator(batch_size)
            bgv = batch_generator(batch_size,False)
            gv =0
            lv = 0
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
                    feed_dict = {input:batch['input'],target :batch['target']}
                    fetches = {
                        "train": model.train,
                        "global_step": sv.global_step,
                            }

                    if should(summary_freq):
                        fetches["summary"] = sv.summary_op

                    if should(progress_freq):
                        fetches["gen_loss_L1"] = model.gen_loss_L1
                        fetches["gen_loss_cross"] = model.gen_loss_cross
		    
		    results = sess.run(fetches,feed_dict = feed_dict)
                    cross_loss+=results['gen_loss_cross']
                    l1_loss+= results['gen_loss_L1']
                    print(l1_loss/bg.batch_index*batch_size,cross_loss/bg.batch_index *batch_size,bg.current_epoch,bg.batch_index,time.time()-start,(time.time()-start)*(bg.batch_len-bg.batch_index)/batch_size+(max_epoch-bg.current_epoch-2)*bg.batch_len/batch_size)
                    if should(summary_freq):
                        print("recording summary")
                        sv.summary_writer.add_summary(results["summary"], (bg.batch_index/bg.batch_size+bg.batch_len/batch_size*bg.current_epoch))
                    if should(save_freq):
                        print("saving model")
                        saver.save(sess, output_dir+"model.ckpt")



