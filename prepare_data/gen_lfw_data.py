# -*- coding: utf-8 -*-
"""

@author: friedhelm
"""
import sys
sys.path.append("../")

from core.MTCNN.mtcnn_detector import MTCNN_Detector
from core.MTCNN.MTCNN_model import Pnet_model,Rnet_model,Onet_model
import numpy as np
import os
from collections import namedtuple
from easydict import EasyDict as edict
from scipy import misc
import cv2
from collections import namedtuple
from core import config
import argparse
from core.tool import preprocess


def arg_parse():
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--input_dir",default=config.lfw_dir,type=str, help='directory to read lfw data')
    parser.add_argument("--output_dir",default=config.lfw_save_dir,type=str, help='path to save lfw_face data')
    parser.add_argument("--image_size",default="112,112",type=str, help='image size')
    
    return parser


def get_DataSet(input_dir, min_images=1):
    
    ret = []
    label = 0
    person_names = []
    for person_name in os.listdir(input_dir):
        person_names.append(person_name)
        person_names = sorted(person_names)
    for person_name in person_names:
        _subdir = os.path.join(input_dir, person_name)
        if not os.path.isdir(_subdir):
            continue
        _ret = []
        for img in os.listdir(_subdir):
            fimage = edict()
            fimage.id = os.path.join(person_name, img)
            fimage.classname = str(label)
            fimage.image_path = os.path.join(_subdir, img)
            fimage.bbox = None
            fimage.landmark = None
            _ret.append(fimage)
        if len(_ret)>=min_images:
            ret += _ret
            label+=1
            
    return ret


def main(args):

    dataset = get_DataSet(args.input_dir)
    print('dataset size', 'lfw', len(dataset))
    
    print('Creating networks and loading parameters')
    
    if(model_name in ["Pnet","Rnet","Onet"]):
        model[0]=Pnet_model
    if(model_name in ["Rnet","Onet"]):
        model[1]=Rnet_model
    if(model_name=="Onet"):
        model[2]=Onet_model    

    detector=MTCNN_Detector(model,model_path,batch_size,factor,min_face_size,threshold)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_filename = os.path.join(args.output_dir, 'lfw_list')

    print('begin to generate')
    with open(output_filename, "w") as text_file:
        nrof_images_total = 0
        nrof = np.zeros( (2,), dtype=np.int32)
        for fimage in dataset:
            if nrof_images_total%100==0:
                print("Processing %d, (%s)" % (nrof_images_total, nrof))
            nrof_images_total += 1
            image_path = fimage.image_path
            if not os.path.exists(image_path):
                print('image not found (%s)'%image_path)
                continue

            try:
                img = cv2.imread(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                _paths = fimage.image_path.split('/')
                a,b = _paths[-2], _paths[-1]
                target_dir = os.path.join(args.output_dir, a)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                target_file = os.path.join(target_dir, b)
                _bbox = None
                _landmark = None
                bounding_boxes, points = detector.detect_single_face(img,False)
                nrof_faces = np.shape(bounding_boxes)[0]
                if nrof_faces>0:
                    det = bounding_boxes[:,0:4]
                    img_size = np.asarray(img.shape)[0:2]
                    bindex = 0
                    if nrof_faces>1:
                        #select the center face according to the characterize of lfw
                        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                        img_center = img_size / 2
                        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)                        
                        bindex = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                    _bbox = bounding_boxes[bindex, 0:4]
                    _landmark = points[bindex, :]
                    nrof[0]+=1
                else:
                    nrof[1]+=1

                warped = preprocess(img, bbox=_bbox, landmark = _landmark, image_size=args.image_size)
                cv2.imwrite(target_file, warped)
                oline = '%d\t%s\t%d\n' % (1,target_file, int(fimage.classname))
                text_file.write(oline)


if __name__=="__main__":
        
    model=[None,None,None]
    #原文参数
    factor=0.79    
    threshold=[0.8,0.8,0.6]
    min_face_size=20
    #原文参数
    batch_size=1
    model_name="Onet"    
    base_dir="."
    
    model_path=[os.path.join(base_dir,"model/MTCNN_model/Pnet_model/Pnet_model.ckpt-20000"),
                os.path.join(base_dir,"model/MTCNN_model/Rnet_model/Rnet_model.ckpt-40000"),
                os.path.join(base_dir,"model/MTCNN_model/Onet_model/Onet_model.ckpt-40000")] 

    args=arg_parse()
    
    #User = namedtuple('User', ['input_dir', 'output_dir', 'image_size'])
    #args = User(input_dir='./data/lfw', output_dir='./data/lfw_face', image_size="112,112")
    
    main(args)
    