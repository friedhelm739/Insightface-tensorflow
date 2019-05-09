# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
import tensorflow as tf
import numpy as np
import os

class Detector(object):
    
    def __init__(self, model, model_path, model_name, batch_size, size_to_predict=128):
        
        if(model_path):
            self.model_name = model_name
            path = model_path.replace("/","\\")
            model_name = path.split("\\")[-1].split(".")[0]
            
            if not os.path.exists(model_path+".meta"):
                raise Exception("%s is not exists"%(model_name))
                
            if(self.model_name == "Pnet"):
                self.size_to_predict = batch_size
                
            graph = tf.Graph()
            with graph.as_default():
                config = tf.ConfigProto(allow_soft_placement=True)  
                config.gpu_options.allow_growth = True 
                self.sess = tf.Session(config = config)
                self.images = tf.placeholder(tf.float32)
                self.label, self.roi, self.landmark = model(self.images,batch_size) 
                saver = tf.train.Saver()
                saver.restore(self.sess,model_path)
     
        self.size_to_predict = size_to_predict

    def predict(self, img):
        """
        used for predict

        Parameters:
        ----------
        img: numpy array 
        
        Returns:
        -------
        pre_label: numpy.array, shape (n,m,2 )
        pre_box: numpy.array, shape (n,m, 4 )        
        pre_land: numpy.array, shape (n,m,10 )
        
            predict
        """    
        pre_labels = []
        pre_boxs = []
        pre_lands = []
        self.batch_size = img.shape[0]

        batch_num = self.batch_size // self.size_to_predict
        left = self.batch_size % self.size_to_predict
            
        if(self.model_name == "Pnet"):
            pre_labels, pre_boxs = self.sess.run([self.label, self.roi], feed_dict={self.images:img})
            pre_lands = np.array([0])
        else:    
            for idx in range(batch_num):
                img_batch = img[idx*self.size_to_predict:(idx+1)*self.size_to_predict]
                pre_label, pre_box, pre_land = self.sess.run([self.label, self.roi, self.landmark], feed_dict={self.images:img_batch})
                pre_labels += list(pre_label)
                pre_boxs += list(pre_box)
                pre_lands += list(pre_land)
            if left :
                img_batch = np.zeros((self.size_to_predict, img.shape[1], img.shape[2],3))
                img_batch[:left,...] = img[-left:]
                pre_label, pre_box, pre_land = self.sess.run([self.label, self.roi, self.landmark], feed_dict={self.images:img_batch})
                pre_labels += list(pre_label)[:left]
                pre_boxs += list(pre_box)[:left]
                pre_lands += list(pre_land)[:left]

        return np.vstack(pre_labels), np.vstack(pre_boxs), np.vstack(pre_lands)