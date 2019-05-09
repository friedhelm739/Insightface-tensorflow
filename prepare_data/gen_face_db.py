# -*- coding: utf-8 -*-
"""

@author: friedhelm
"""
import sys
sys.path.append("../")

import os
import cv2
import numpy as np
from recognizer.arcface_recognizer import Arcface_recognizer
import pymysql
import argparse
from core import config
from collections import namedtuple


def arg_parse():
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--input_dir", default=config.custom_dir, type=str, help='directory to read lfw data')
    parser.add_argument("--output_dir", default=config.embds_save_dir, type=str, help='directory to save embds')
    parser.add_argument("--arc_model_name", default=config.arc_model_name, type=str, help='arcface model name')
    parser.add_argument("--arc_model_path", default=config.arc_model_path, type=str, help='directory to read arcface model')
    parser.add_argument("--mtcnn_model_path", default=config.mtcnn_model_path, type=str, help='directory to read MTCNN model')
    
    return parser


def main(args):
    
    recognizer = Arcface_recognizer(args.arc_model_name, args.arc_model_path, args.mtcnn_model_path)
    label = 0
    input_lists = os.listdir(args.input_dir)
    print(input_lists)
    embds = np.zeros((len(input_lists), 1, 512))
    l2_embds = np.zeros((len(input_lists), 1, 512))
    for idx, person_name in enumerate(input_lists):
        
        _subdir = os.path.join(args.input_dir, person_name)
        
        if not os.path.isdir(_subdir):
            continue

        num = 0.0
        print(person_name)
        embd_tp_save = np.zeros((1,512))
        for img_name in os.listdir(_subdir):
            
            image_path = os.path.join(_subdir, img_name)
            img = cv2.imread(image_path)
            embd, bounding_boxes = recognizer.get_embd(img)
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
        
        embd_tp_save /= num
        
        if(embd_tp_save == np.zeros((1,512))):
            print(person_name+"has no face to detect")
            continue
            
        embds[idx,:] = embd_tp_save
        l2_embds[idx,:] = embd_tp_save/np.linalg.norm(embd_tp_save, axis=1, keepdims=True)
        # num represents numbers of embd,label represents the column number
        
        sql = "INSERT INTO face_data(FaceName,EmbdNums, ColumnNum) VALUES ('%s', %.4f, %d)"%(person_name, num, label)
        try:
            cursor.execute(sql)
            db.commit()
        except:
            db.rollback()
            raise Exception('''mysql "INSERT" action error in label %d"'''%(label))
        label += 1
        
    np.save(args.output_dir+"/l2_embds.npy", l2_embds)    
    np.save(args.output_dir+"/embds.npy", embds)

if __name__ == "__main__":

    db= pymysql.connect(host="localhost",user="root",password="dell",port=3306,charset="utf8")
    cursor = db.cursor()
    cursor.execute("DROP DATABASE IF EXISTS face_db")
    cursor.execute("CREATE DATABASE IF NOT EXISTS face_db")
    cursor.execute("use face_db;")
    cursor.execute("alter database face_db character set gbk;")
    cursor.execute("CREATE TABLE IF NOT EXISTS face_data(FaceName VARCHAR(50), EmbdNums INT, ColumnNum INT);")

    #User = namedtuple('User', ['input_dir', 'arc_model_name', 'arc_model_path', 'mtcnn_model_path', "output_dir"])
    #args = User(input_dir=config.custom_dir, arc_model_name=config.arc_model_name, arc_model_path=config.arc_model_path, mtcnn_model_path=config.mtcnn_model_path, output_dir=config.embds_save_dir)
    
    args = arg_parse()
    main(args)
    
    cursor.close()