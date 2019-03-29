# -*- coding: utf-8 -*-
"""

@author: friedhelm
"""
import sys
sys.path.append("../")

import tensorflow as tf
from train_tool import arcface_loss,read_single_tfrecord
from core import Arcface_model,config
import time
import os

def train(image,label,train_phase_dropout,train_phase_bn):
    
    train_images,train_labels=read_single_tfrecord(addr,batch_size,img_size)   
    train_images = tf.identity(train_images, 'input_images')
    train_labels = tf.identity(train_labels, 'labels')
    
    net, end_points = Arcface_model.get_embd(image, train_phase_dropout, train_phase_bn,config.model_params)
            
    logit=arcface_loss(net,label,config.s,config.m)
    arc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy(logits = logit , labels = label))
    L2_loss=tf.reduce_sum(tf.get_collection(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
    
    with tf.name_scope('loss'):
        train_loss=arc_loss+L2_loss
        tf.summary.scalar('train_loss',train_loss) 
        
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    
    scale = int(512.0/batch_size)
    lr_steps = [scale*s for s in config.lr_steps]
    lr_values = [v/scale for v in config.lr_values]
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=lr_values, name='lr_schedule')
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=config.momentum).minimize(train_loss)    
    
    with tf.name_scope('accuracy'):
        train_accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logit)),label),tf.float32))
        tf.summary.scalar('train_accuracy',train_accuracy) 
        
    saver=tf.train.Saver(max_to_keep=10)
    merged=tf.summary.merge_all() 
    
    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(),
                  tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        

        writer_train=tf.summary.FileWriter("../model/%s"%(model_name),sess.graph)
        try:
            for i in range(1,train_step):
                
                image_batch,label_batch=sess.run([train_images,train_labels])
                
                sess.run([train_op,inc_op],feed_dict={image:image_batch,label:label_batch,train_phase_dropout:True,train_phase_bn:True})
                if(i%100==0):
                    summary=sess.run(merged,feed_dict={image:image_batch,label:label_batch,train_phase_dropout:True,train_phase_bn:True})
                    writer_train.add_summary(summary,i) 
                if(i%1000==0):
                    print('次数',i)    
                    print('train_accuracy',sess.run(train_accuracy,feed_dict={image:image_batch,label:label_batch,train_phase_dropout:True,train_phase_bn:True}))
                    print('train_loss',sess.run(train_loss,{image:image_batch,label:label_batch,train_phase_dropout:True,train_phase_bn:True}))                    
                    print('time',time.time()-begin)
                    if(i%10000==0):
                        saver.save(sess,os.path.join(model_path,model_name),global_step=i)
        except  tf.errors.OutOfRangeError:
            print("finished")
        finally:
            coord.request_stop()
            writer_train.close()
        coord.join(threads)

def main():
    
     with tf.name_scope('input'):
        image=tf.placeholder(tf.float32,[batch_size,img_size,img_size,3],name='image')
        label=tf.placeholder(tf.int32,[batch_size],name='label')
        train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_dropout')
        train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_bn') 
        
     train(image,label,train_phase_dropout,train_phase_bn)


if __name__ == "__main__":
    
    img_size=config.img_size
    batch_size=config.batch_size
    addr=config.addr
    model_name=config.model_name
    train_step=config.train_step
    model_path=config.model_path
    
    begin=time.time()
    
    main()
    

    