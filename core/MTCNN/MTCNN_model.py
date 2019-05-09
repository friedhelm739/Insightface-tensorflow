# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
import tensorflow as tf

def prelu(inputs):

    with tf.variable_scope('prelu'):
        alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
        pos = tf.nn.relu(inputs)
        neg = alphas * (inputs-abs(inputs))*0.5
    
    return pos + neg


def conv2d(_input,name,conv_size,conv_stride,bias_size,pad,activation='prelu'):
    
    regularizer=tf.contrib.layers.l2_regularizer(0.0005)
    with tf.variable_scope(name):
        weight=tf.get_variable('weight',conv_size,initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias=tf.get_variable('bias',bias_size,initializer=tf.constant_initializer(0.0))
        weight_loss=regularizer(weight)
        tf.add_to_collection('loss',weight_loss)
        conv=tf.nn.conv2d(_input,weight,strides=conv_stride,padding=pad)
        he=tf.nn.bias_add(conv,bias)
        relu=tf.cond(tf.equal(activation,'prelu'),lambda:prelu(he),lambda:tf.cond(tf.equal(activation,'softmax'),lambda:tf.nn.softmax(he),lambda:he),name='output')     
    
    return relu


def fc2d(_input,name,fc_size,bias_size,activation='prelu'):
    
    regularizer=tf.contrib.layers.l2_regularizer(0.0005)
    with tf.variable_scope(name):
        weight=tf.get_variable('weight',fc_size,initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias=tf.get_variable('bias',bias_size,initializer=tf.constant_initializer(0.0))
        weight_loss=regularizer(weight)
        tf.add_to_collection('loss',weight_loss)
        he=tf.nn.bias_add(tf.matmul(_input,weight,name='matmul'),bias)
        relu=tf.cond(tf.equal(activation,'prelu'),lambda:prelu(he),lambda:tf.cond(tf.equal(activation,'softmax'),lambda:tf.nn.softmax(he),lambda:he))     
    
    return relu


def pool(_input,name,kernal_size,kernal_stride,pad):
    
    with tf.variable_scope(name):  
        pool=tf.nn.max_pool(_input,ksize=kernal_size,strides=kernal_stride,padding=pad)  
        
    return pool


def Pnet_model(x,batch_size):
    
    conv_1=conv2d(x,'conv_1',[3,3,3,10],[1,1,1,1],[10],'VALID')
    pool_1=pool(conv_1,'pool_1',[1,3,3,1],[1,2,2,1],'SAME')     
    conv_2=conv2d(pool_1,'conv_2',[3,3,10,16],[1,1,1,1],[16],'VALID')
    conv_3=conv2d(conv_2,'conv_3',[3,3,16,32],[1,1,1,1],[32],'VALID')
    
    face_label=conv2d(conv_3,'face_label',[1,1,32,2],[1,1,1,1],[2],'VALID','softmax')
    bounding_box=conv2d(conv_3,'bounding_box',[1,1,32,4],[1,1,1,1],[4],'VALID','None')
    landmark_local=conv2d(conv_3,'landmark_local',[1,1,32,10],[1,1,1,1],[10],'VALID','None')      
        
    return face_label, bounding_box ,landmark_local


def Rnet_model(x,batch_size):
    
    conv_1=conv2d(x,'conv_1',[3,3,3,28],[1,1,1,1],[28],'VALID')
    pool_1=pool(conv_1,'pool_1',[1,3,3,1],[1,2,2,1],'SAME')
    conv_2=conv2d(pool_1,'conv_2',[3,3,28,48],[1,1,1,1],[48],'VALID')
    pool_2=pool(conv_2,'pool_2',[1,3,3,1],[1,2,2,1],'VALID')    
    conv_3=conv2d(pool_2,'conv_3',[2,2,48,64],[1,1,1,1],[64],'VALID')
    
    resh1 = tf.reshape(conv_3, [batch_size,3*3*64], name='resh1')
    
    fc_1=fc2d(resh1,'fc_1',[3*3*64,128],[128])    

    face_label=fc2d(fc_1,'face_label',[128,2],[2],'softmax')
    bounding_box=fc2d(fc_1,'bounding_box',[128,4],[4],'None')
    landmark_local=fc2d(fc_1,'landmark_local',[128,10],[10],'None')

    return face_label, bounding_box ,landmark_local


def Onet_model(x,batch_size):
    
    conv_1=conv2d(x,'conv_1',[3,3,3,32],[1,1,1,1],[32],'VALID')
    pool_1=pool(conv_1,'pool_1',[1,3,3,1],[1,2,2,1],'SAME')
    conv_2=conv2d(pool_1,'conv_2',[3,3,32,64],[1,1,1,1],[64],'VALID')
    pool_2=pool(conv_2,'pool_2',[1,3,3,1],[1,2,2,1],'VALID')    
    conv_3=conv2d(pool_2,'conv_3',[3,3,64,64],[1,1,1,1],[64],'VALID')
    pool_3=pool(conv_3,'pool_3',[1,2,2,1],[1,2,2,1],'SAME')  
    conv_4=conv2d(pool_3,'conv_4',[2,2,64,128],[1,1,1,1],[128],'VALID')   

    resh1 = tf.reshape(conv_4, [batch_size,3*3*128], name='resh1')
    
    fc_1=fc2d(resh1,'fc_1',[3*3*128,256],[256])
           
    face_label=fc2d(fc_1,'face_label',[256,2],[2],'softmax')
    bounding_box=fc2d(fc_1,'bounding_box',[256,4],[4],'None')
    landmark_local=fc2d(fc_1,'landmark_local',[256,10],[10],'None')

    return face_label, bounding_box ,landmark_local