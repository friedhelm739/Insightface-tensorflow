# -*- coding: utf-8 -*-
"""

@author: friedhelm
"""
import sys
sys.path.append("../")

from core import Arcface_model,config
import os
import tensorflow as tf
import numpy as np


class Recognizer():
    
    def __init__(self, arc_model_name, arc_model_path, size_to_predict, image_size):
        
        #prevent recurrent reading
        self.arc_model_name = arc_model_name
        path = arc_model_path.replace("/","\\")
        model_name = path.split("\\")[-1]
        if not os.path.exists(arc_model_path+".meta"):
            raise Exception("%s is not exists"%(model_name))
   
        graph=tf.Graph()
        with graph.as_default():
            tf_config = tf.ConfigProto(allow_soft_placement=True)
            tf_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config = tf_config)
            
            self.image = tf.placeholder(tf.float32,[None,image_size[0],image_size[1],3], name='image')
            self.train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_dropout')
            self.train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_bn') 
            self.net, _ = Arcface_model.get_embd(self.image, self.train_phase_dropout, self.train_phase_bn, config.model_params)
 
            saver=tf.train.Saver()
            saver.restore(self.sess,arc_model_path)            

        self.size_to_predict = size_to_predict
        
        
    def predict(self, img):
            
        emdbs = []
        batch_size = img.shape[0]
        batch_num = int(batch_size/self.size_to_predict)
        left = int(batch_size%self.size_to_predict)

        for idx in range(batch_num):
            img_batch = img[idx*self.size_to_predict:(idx+1)*self.size_to_predict]
            emdb = self.sess.run(self.net,feed_dict={self.image:img, self.train_phase_dropout:config.eval_dropout_flag, self.train_phase_bn:config.eval_bn_flag}) 
            emdbs += list(emdb)

        if left :
            img_batch = np.zeros((self.size_to_predict, img.shape[1], img.shape[2],3))
            img_batch[:left,...] = img[-left:]
            emdb = self.sess.run(self.net,feed_dict={self.image:img, self.train_phase_dropout:config.eval_dropout_flag, self.train_phase_bn:config.eval_bn_flag}) 
            emdbs += list(emdb)[:left]

        return emdbs
