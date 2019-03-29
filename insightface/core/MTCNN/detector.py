# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
import tensorflow as tf
import numpy as np
import os

class Detector(object):
    
    def __init__(self,model,model_path,model_name,batch_size):
        
        if(model_path):
            self.model_name=model_name
            path=model_path.replace("/","\\")
            model_name=path.split("\\")[-1].split(".")[0]
            if not os.path.exists(model_path+".meta"):
                raise Exception("%s is not exists"%(model_name))
                
            graph=tf.Graph()
            with graph.as_default():
                config = tf.ConfigProto(allow_soft_placement=True)  
                config.gpu_options.allow_growth = True 
                self.sess=tf.Session(config = config)
                self.images=tf.placeholder(tf.float32)
                self.label,self.roi,self.landmark=model(self.images,batch_size) 
                saver=tf.train.Saver()
                saver.restore(self.sess,model_path)
     
        
    def predict(self,img):
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
        pre_land=np.array([0])
        if(self.model_name=="Onet"):
            pre_label,pre_box,pre_land=self.sess.run([self.label,self.roi,self.landmark],feed_dict={self.images:img})
        else:
            pre_label,pre_box=self.sess.run([self.label,self.roi],feed_dict={self.images:img}) 
            
        return np.vstack(pre_label),np.vstack(pre_box),np.vstack(pre_land)