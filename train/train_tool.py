# -*- coding: utf-8 -*-
"""

@author: friedhelm
"""
import sys
sys.path.append("../")

import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
from core import config

def arcface_loss(inputs,labels,s,m):
    
    with tf.name_scope("arcface_loss"):
        
        weight = tf.get_variable("loss_wight",[inputs.get_shape().as_list()[-1], config.class_num],
                                               initializer = tf.contrib.layers.xavier_initializer(),
                                               regularizer=slim.l2_regularizer(config.model_params["weight_decay"]))
        
        inputs = tf.nn.l2_normalize(inputs, axis=1)
        weight = tf.nn.l2_normalize(weight, axis=0)
        
        sin_m = math.sin(m)
        cos_m = math.cos(m)
        mm = sin_m * m
        threshold = math.cos(math.pi - m)
        
        cos_theta = tf.matmul(inputs,weight,name="cos_theta")
        sin_theta = tf.sqrt(tf.subtract(1. , tf.square(cos_theta)))
        
        cos_theta_m = s * tf.subtract(tf.multiply(cos_theta , cos_m) , tf.multiply(sin_theta , sin_m))
        keep_val = s * (cos_theta - mm)
        
        cond_v = cos_theta - threshold
        cond= tf.cast(tf.nn.relu(cond_v),dtype=tf.bool)
        cos_theta_m_keep = tf.where(cond , cos_theta_m , keep_val)
        
        mask = tf.one_hot(labels , config.class_num)
        inv_mask = tf.subtract(1., mask)
        
        output = tf.add(tf.multiply(mask , cos_theta_m_keep) , tf.multiply(inv_mask , s * cos_theta) , name="arcface_loss")
        
    
    return output


def read_single_tfrecord(addr,_batch_size,shape):
    
    filename_queue = tf.train.string_input_producer([addr],shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 

    features = tf.parse_single_example(serialized_example,
                                   features={
                                   'img':tf.FixedLenFeature([],tf.string),
                                   'label':tf.FixedLenFeature([],tf.int64),                                   
                                   })
    img=tf.decode_raw(features['img'],tf.uint8)
    label=tf.cast(features['label'],tf.int32)
    img = tf.reshape(img, [shape,shape,3])
    img = augmentation(img)
    img=(tf.cast(img,tf.float32)-127.5)/128 
    min_after_dequeue = 10000
    batch_size = _batch_size
    capacity = min_after_dequeue + 10 * batch_size
    image_batch, label_batch= tf.train.shuffle_batch([img,label], 
                                                        batch_size=batch_size, 
                                                        capacity=capacity, 
                                                        min_after_dequeue=min_after_dequeue,
                                                        num_threads=7)  


    label_batch = tf.reshape(label_batch, [batch_size])

    
    return image_batch, label_batch


def augmentation(image):

    image = tf.image.random_flip_left_right(image)
    
    return image


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads