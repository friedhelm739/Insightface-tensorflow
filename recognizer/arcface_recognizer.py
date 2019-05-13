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
import cv2


class Arcface_recognizer():
    
    def __init__(self, 
                 arc_model_name, 
                 arc_model_path, 
                 mtcnn_model_path, 
                 batch_size_to_recognize_one_time=128,
                 host="localhost",
                 user="root",
                 password="dell",
                 port=3306,
                 database="face_db",
                 table="face_data",
                 image_size=[112,112], 
                 batch_size=1, 
                 mtcnn_model_name="Onet", 
                 factor=0.79, 
                 min_face_size=10, 
                 threshold=[0.8,0.8,0.6],
                 raw_face_data_name="raw_embds",
                 l2_face_data_name="l2_embds"):
        
        model=[None,None,None]
        if(mtcnn_model_name in ["Pnet","Rnet","Onet"]):
            model[0]=MTCNN_model.Pnet_model
        if(mtcnn_model_name in ["Rnet","Onet"]):
            model[1]=MTCNN_model.Rnet_model
        if(mtcnn_model_name=="Onet"):
            model[2]=MTCNN_model.Onet_model 
          
        self.img_size_list = image_size
        self.face_detector = mtcnn_detector.MTCNN_Detector(model,mtcnn_model_path,batch_size,factor,min_face_size,threshold)
        self.recognizer = recognizer.Recognizer(arc_model_name, arc_model_path, batch_size_to_recognize_one_time, self.img_size_list)
        self.image_size = str(image_size[0]) + "," + str(image_size[1])
        self.database = database
        self.table = table
        self.db = pymysql.connect(host=host, user=user, password=password, port=port, charset="utf8")
        self.cursor = self.db.cursor()
        self.cursor.execute("CREATE DATABASE IF NOT EXISTS %s"%(database))
        self.cursor.execute("USE %s;"%(database))
        self.cursor.execute("ALTER DATABASE %s character SET gbk;"%(database))
        self.cursor.execute("CREATE TABLE IF NOT EXISTS %s(FaceName VARCHAR(20), EmbdNums INT, ColumnNum INT);"%(table))
        self.load_raw_dir = os.path.join(config.embds_save_dir, "%s.npy"%(raw_face_data_name)) 
        self.load_l2_dir = os.path.join(config.embds_save_dir, "%s.npy"%(l2_face_data_name)) 
        self.l2_face_data_name = l2_face_data_name     
        self.raw_face_data_name = raw_face_data_name  
        
    
    def align_face(self, img, _bbox, _landmark):
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
            warped[i,:] = preprocess(img, bbox=_bbox[i], landmark=_landmark[i], image_size=self.image_size)
        
        return warped

        
    def get_embd(self, img, **kwargs):
        """
        used for getting embeddings
        
        Parameters:
        ----------
            img : numpy.array
            
           kwargs:     
                detect_method : str
                print_sign : bool
                if_align : bool
                
        Returns:
        -------
            embd: list
            bounding_boxes: list (x1,y1,x2,y2）
            get_embd
        """                       
        if (kwargs.kwargs.get('detect_method', "single")=="single"):
            bounding_boxes, points = self.face_detector.detect_single_face(img,kwargs.get('print_sign', False))
        else: 
            bounding_boxes, points = self.face_detector.detect_face(img,kwargs.get('print_sign', False))
            
        if np.shape(bounding_boxes)[0] == 0:
            return None, None
        
        if kwargs.get('if_align', True):
            warped = self.align_face(img, bounding_boxes, points)
        else :
            warped = np.zeros((len(bounding_boxes), self.img_size_list[0], self.img_size_list[1], 3))
            for i in range(len(bounding_boxes)):
                
                img_face = img[bounding_boxes[i][1]:bounding_boxes[i][3], bounding_boxes[i][0]:bounding_boxes[i][2], :]
                img_face = cv2.resize(img_face, (self.img_size_list[0], self.img_size_list[1]))
                warped[i] = img_face
        
        warped = (warped-127.5)/128 
        embd = self.recognizer.predict(warped)
            
        return embd, bounding_boxes
    

    def recognize(self, img, **kwargs):
        """
        used for recognizing
        
        Parameters:
        ----------
            img : numpy.array
            
            kwargs:
                threathold: int
                recognize_mode :str
                detect_method : str
                print_sign : bool
                if_align : bool
                
        Returns:
        -------
            names: list 
            bounding_boxes :list (x1,y1,x2,y2）
            
            recognize
        """    
        names = []
        embds, bounding_boxes = self.get_embd(img, kwargs)

        if not os.path.exists(self.load_l2_dir):
            raise Exception(''' Face Data "%s.npy" Does Not Exists '''%(self.l2_face_data_name))
            
        embds_db = np.load(self.load_l2_dir)
        embds_db = np.squeeze(embds_db) 
        
        
        for idx in range(len(embds)):
            
            embd = embds[idx]

            embd = np.reshape(embd,(1,config.model_params["embd_size"]))
            embd = embd/np.linalg.norm(embd, axis=1, keepdims=True)

            diff = np.subtract(embds_db, embd)
            dist = np.sum(np.square(diff),1)

            column = np.where(dist < kwargs.get('threathold', 1))[0]
            
            if(len(column) == 1 or kwargs.get('recognize_mode', "single") != "single"):
                
                for i in range(len(column)):
                    self.cursor.execute('''SELECT FaceName from %s WHERE ColumnNum = %d;'''%(self.table, column[i]))
                    name_data = self.cursor.fetchall()
                    name = name_data[0][0]
                    names.append(name)
            else :
                names.append(None)

        return names, bounding_boxes
    
    
    def add_customs(self, input_dir, **kwargs):
        """
        used for adding customs' data and adding embds of existing customs
        
        Parameters:
        ----------
            input_dir : str
            
            kwargs:
                detect_method : str
                print_sign : bool
                if_align : bool
                replace_flag :bool    
            
        Returns:
        -------
             
        
            add_customs
        """     
        if not os.path.exists(input_dir):
            raise Exception("input dir does not exists")

        input_lists = os.listdir(input_dir)

        #find the missing label by adding auto_increment primary key into table
        self.cursor.execute("alter table %s add Crement int;"%(self.table))
        self.cursor.execute("alter table %s change crement crement int not null auto_increment primary key;"%(self.table))
        self.cursor.execute("SELECT Crement-1 FROM %s WHERE Crement-1 NOT IN (SELECT ColumnNum FROM %s) ;"%(self.table,self.table))
        ColumnNum_data_loss = self.cursor.fetchall()  

        if not os.path.exists(config.embds_save_dir):
            os.makedirs(config.embds_save_dir)
            
        try:
            #there are two conditions 
            if not os.path.exists(self.load_l2_dir):
                label = list(range(len(input_lists)))
                embds_db = np.zeros((len(input_lists)+1, 1, 512))
                embds = np.zeros((len(input_lists)+1, 1, 512))
            else:
                embds_db = np.load(self.load_l2_dir) 
                embds = np.load(self.load_raw_dir)
                row,_ ,_= embds_db.shape          
                label_data = [_[0] for _ in ColumnNum_data_loss]
                expand_label = list(range(row,row + len(input_lists) - len(label_data)))
                label = label_data + expand_label
                np.row_stack((embds_db, np.zeros((len(expand_label)+1, 1, 512))))
                np.row_stack((embds, np.zeros((len(expand_label)+1, 1, 512))))
                
            invalid_num =0   
            idx = 0
            for person_name in input_lists:
                
                _subdir = os.path.join(input_dir, person_name)
                
                if not os.path.isdir(_subdir):
                    continue
        
                num = 0.0
                embd_tp_save = np.zeros((1,512))
                for img_name in os.listdir(_subdir):
                    
                    image_path = os.path.join(_subdir, img_name)
                    img=cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),-1)
                    
                    if(min(img.shape[:2])<12):
                        raise Exception("%s size can not less than 12*12"%(img_name))
                        
                    embd, bounding_boxes = self.get_embd(img, kwargs)
                    if embd is None:
                        continue
                        
                    nrof_faces = np.shape(bounding_boxes)[0]
                    if nrof_faces>1:
                        #select the center face according to the characterize of lfw
                        det = bounding_boxes[:,0:4]
                        img_size = np.asarray(img.shape)[0:2]
                        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                        img_center = img_size / 2
                        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)                        
                        bindex = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                        embd = embd[bindex]
         
                    num += 1
                    embd_tp_save += embd
                
                if(all((embd_tp_save==np.zeros((1,512)))[0])):
                    print(person_name+" has no face to detect")
                    invalid_num+=1
                    continue
                else:
                    embd_tp_save /= num   
                    
                self.cursor.execute("select count(*) from %s where FaceName ='%s';"%(self.table, person_name))
                name_exists = self.cursor.fetchall()[0][0]
                
                if name_exists:
                    
                    invalid_num+=1
                    self.cursor.execute("select EmbdNums,ColumnNum from %s where FaceName ='%s';"%(self.table, person_name))
                    EmbdNums,ColumnNum = self.cursor.fetchall()[0]  
                    
                    if not kwargs.get('replace_flag', False):
                        embd_all = (embds[ColumnNum,:] * EmbdNums) + (embd_tp_save * num)
                        embds[ColumnNum,:] = embd_all/(EmbdNums + num)
                        embds_db[ColumnNum,:] = embd_all/np.linalg.norm(embd_all, axis=1, keepdims=True)
                    else:
                        embds_db[ColumnNum,:] = embd_tp_save/np.linalg.norm(embd_tp_save, axis=1, keepdims=True)
                        embds[ColumnNum,:] = embd_tp_save
                        EmbdNums = 0
                        
                    try:
                        self.cursor.execute("UPDATE %s SET EmbdNums = %d WHERE FaceName = '%s';"%(self.table, EmbdNums + num, person_name))
                        self.db.commit()
                    except:
                        self.db.rollback()
                        raise Exception('''mysql "UPDATE" action error in FaceName %d"'''%(person_name))
                
                else:    
                    embds[label[idx],:] = embd_tp_save
                    embds_db[label[idx],:] = embd_tp_save/np.linalg.norm(embd_tp_save, axis=1, keepdims=True)
                    
                    # num represents numbers of embd,label represents the column number
                    try:
                        self.cursor.execute("INSERT INTO face_data(FaceName,EmbdNums, ColumnNum) VALUES ('%s', %.4f, %d)"%(person_name, num, label[idx]))
                        self.db.commit()
                    except:
                        self.db.rollback()
                        raise Exception('''mysql "INSERT" action error in label %d"'''%(label[idx]))
                    
                    idx += 1
            np.save(self.load_l2_dir, embds_db[:-1-invalid_num,])  
            np.save(self.load_raw_dir, embds[:-1-invalid_num,]) 

            self.cursor.execute("alter table %s drop column Crement;"%(self.table))
            
        except:
            
            self.db.rollback()
            self.cursor.execute("alter table %s drop column Crement;"%(self.table))
            raise Exception("unexpected error occur, Database rollback")
            
        return 


    def add_embds(self,input_dir, **kwargs):
        """
        used for adding embeddings
        
        Parameters:
        ----------
            input_dir : str
            
            kwargs:
                detect_method : str
                print_sign : bool
                if_align : bool
                
        Returns:
        -------
            add_customs : function
        
            add_embds
        """        
        return self.add_customs(input_dir, kwargs)
    
    
    def update_customs(self, input_dir, **kwargs):
        """
        used for replacing customs' old data
        
        Parameters:
        ----------
            input_dir : str
            
            kwargs:
                detect_method : str
                print_sign : bool
                if_align : bool
                replace_flag: bool
                
        Returns:
        -------
            list 
        
            add_customs
        """    
        return self.add_customs(input_dir, kwargs)   
     
        
    def del_customs(self, person_name_list):
        """
        used for adding customs
        
        Parameters:
        ----------
            person_name_list : str

                
        Returns:
        -------
        
            add_customs
        """            
        for idx in range(len(person_name_list)):
            try:
                self.cursor.execute("delete from %s where FaceName='%s';"%(self.table, person_name_list[idx]))
                self.db.commit()
            except:
                self.db.rollback()
                raise Exception('''mysql "DELETE" action error in FaceName %d"'''%(person_name_list[idx]))
        
        self.cursor.execute("SELECT * FROM %s ;"%(self.table))
        is_empty = self.cursor.fetchall()
        #prevent Memory leak and column error
        if(len(is_empty)==0):
            if os.path.exists(self.load_raw_dir):
                os.remove(self.load_raw_dir)
            if os.path.exists(self.load_l2_dir):    
                os.remove(self.load_l2_dir)
        return 
    
    
    def close_db(self):
        """
        used for closing database
        
        Parameters:
        ----------

                
        Returns:
        -------
        
            add_customs
        """            
        self.db.close()
        
        return 