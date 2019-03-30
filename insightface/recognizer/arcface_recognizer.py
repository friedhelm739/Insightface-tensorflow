# -*- coding: utf-8 -*-
"""

@author: friedhelm
"""
import sys
sys.path.append("../")

from . import Recognizer
from core import config
from core.MTCNN import mtcnn_detector,MTCNN_model


class Arcface_recognizer():
    
    def __init__(self, arc_model_name, arc_model_path, mtcnn_model_path, batch_size=1, mtcnn_model_name="Onet", factor=0.79, min_face_size=10, threshold=[0.8,0.8,0.6]):
        
        model=[None,None,None]
        if(mtcnn_model_name in ["Pnet","Rnet","Onet"]):
            model[0]=MTCNN_model.Pnet_model
        if(mtcnn_model_name in ["Rnet","Onet"]):
            model[1]=MTCNN_model.Rnet_model
        if(mtcnn_model_name=="Onet"):
            model[2]=MTCNN_model.Onet_model 
            
        self.face_detector = mtcnn_detector.MTCNN_Detector(model,mtcnn_model_path,batch_size,factor,min_face_size,threshold)
        self.recognizer = Recognizer(arc_model_name, arc_model_path)
        
        
    def recognize(self, img, detect_method="single", print_sign=False):
        
        if(detect_method=="single"):
            img = self.face_detector.detect_single_face(img,print_sign)
        else: 
            img = self.face_detector.detect_face(img,print_sign)
            
        emdb = self.recognizer.predict(img)
            
        return emdb