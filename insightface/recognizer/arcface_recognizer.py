# -*- coding: utf-8 -*-
"""

@author: friedhelm
"""
import sys
sys.path.append("../")

from . import recognizer
from core import config
from core.MTCNN import mtcnn_detector,MTCNN_model
from core.tool import preprocess
import numpy as np
import pymysql
import os


class Arcface_recognizer():
    
    def __init__(self, 
                 arc_model_name, 
                 arc_model_path, 
                 mtcnn_model_path, 
                 size_to_predict=128,
                 host="localhost",
                 user="root",
                 password="dell",
                 port=3306,
                 database="face_db",
                 image_size=[112,112], 
                 batch_size=1, 
                 mtcnn_model_name="Onet", 
                 factor=0.79, 
                 min_face_size=10, 
                 threshold=[0.8,0.8,0.6]):
        
        model=[None,None,None]
        if(mtcnn_model_name in ["Pnet","Rnet","Onet"]):
            model[0]=MTCNN_model.Pnet_model
        if(mtcnn_model_name in ["Rnet","Onet"]):
            model[1]=MTCNN_model.Rnet_model
        if(mtcnn_model_name=="Onet"):
            model[2]=MTCNN_model.Onet_model 
          
        self.img_size_list = image_size
        self.face_detector = mtcnn_detector.MTCNN_Detector(model,mtcnn_model_path,batch_size,factor,min_face_size,threshold)
        self.recognizer = recognizer.Recognizer(arc_model_name, arc_model_path, size_to_predict, self.img_size_list)
        self.image_size = str(image_size[0]) + "," + str(image_size[1])
        self.database = database
        db = pymysql.connect(host=host, user=user, password=password, port=port, charset="utf8" )
        self.cursor = db.cursor()
        self.cursor.execute("USE %s;"%(database))
        self.cursor.execute("ALTER DATABASE %s character SET gbk;"%(database))   
        

    def align_face(self, img, _bbox, _landmark, if_align=True):
        """
        used for aligning face
    
        Parameters:
        ----------
            img : numpy.array 
            _bbox : numpy.array shape=(n,1,4)
            _landmark : numpy.array shape=(n,5,2)
            if_align : bool
            
        Returns:
        -------
            numpy.array 
        
            align_face
        """         
        num = np.shape(_bbox)[0]
        warped = np.zeros((num,self.img_size_list[0],self.img_size_list[1],3))
        
        for i in range(num):
            warped[i,:] = preprocess(img, bbox=_bbox[i], landmark=_landmark[i], image_size=self.image_size, align=if_align)
        
        return warped

        
    def get_embd(self, img, detect_method="single", print_sign=False, if_align=True):
        """
        used for getting embeddings
        
        Parameters:
        ----------
            img : numpy.array
            detect_method : str
            print_sign : bool
            if_align : bool
                
        Returns:
        -------
            numpy.array 
        
            get_embd
        """                       
        if(detect_method=="single"):
            bounding_boxes, points = self.face_detector.detect_single_face(img,print_sign)
        else: 
            bounding_boxes, points = self.face_detector.detect_face(img,print_sign)
            
        if np.shape(bounding_boxes)[0] == 0:
            return None

        warped = self.align_face(img, bounding_boxes, points, if_align)
        embd = self.recognizer.predict(warped)
            
        return embd, bounding_boxes
    

    def recognize(self, img, threathold=1.5, detect_method="single", print_sign=False, if_align=True):
        """
        used for recognizing
        
        Parameters:
        ----------
            img : numpy.array
            detect_method : str
            print_sign : bool
            if_align : bool
                
        Returns:
        -------
            list 
        
            recognize
        """    
        names = []
        embds, bounding_boxes = self.get_embd(img, detect_method, print_sign, if_align)
        embds_db = np.load(config.embds_save_dir+"/l2_embds.npy")
        embds_db = np.squeeze(embds_db) 
        
        for idx in range(len(embds)):
            
            embd = embds[idx]
            embd = np.reshape(embd,(1,config.model_params["embd_size"]))
            embd = embd/np.linalg.norm(embd, axis=1, keepdims=True)

            diff = np.subtract(embds_db, embd)
            dist = np.sum(np.square(diff),1)
            
            column = np.where(dist<threathold)[0]
            print(column)
            
            if(len(column) == 1):
                self.cursor.execute('''SELECT FaceName from %s WHERE ColumnNum = %d;'''%(self.database, column[0]))
                data = self.cursor.fetchall()
                name = data[0][0]
                names.append(name)
            else :
                names.append(None)

        return names, bounding_boxes
    
    
    def add_customs(self, input_dir):
        """
        used for adding customs
        
        Parameters:
        ----------
            input_dir : str

                
        Returns:
        -------
            list 
        
            add_customs
        """       
    
    
    def add_embds(self):
        """
        used for adding embeddings
        
        Parameters:
        ----------
            img : numpy.array
            detect_method : str
            print_sign : bool
            if_align : bool
                
        Returns:
        -------
            list 
        
            add_embds
        """        
    
    
    