# -*- coding: utf-8 -*-
"""

@author: friedhelm
"""
import tensorflow as tf
import mxnet as mx
import os
import io
import numpy as np
import cv2
import time
from scipy import misc
import argparse
from core import config


def arg_parse():
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--read_dir",default=config.mxdata_dir,type=str, help='directory to read data')
    parser.add_argument("--save_dir",default=config.tfrecord_dir,type=str, help='path to save TFRecord file')
    
    return parser


def main():
    
    with tf.python_io.TFRecordwriter(save_dir) as writer:
        
        idx_path = os.path.join(read_dir, 'train.idx')
        bin_path = os.path.join(read_dir, 'train.rec')
        imgrec = mx.recordio.MXIndexedRecordIO(idx_path, bin_path, 'r')
        s = imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        imgidx = list(range(1, int(header.label[0])))
        labels = []
        for i in imgidx:
            img_info = imgrec.read_idx(i)
            header, img = mx.recordio.unpack(img_info)
            label = int(header.label)
            labels.append(label)
            img = io.BytesIO(img)
            img = misc.imread(img).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #img = cv2.resize(img, (112,112))
            img_raw = img.tobytes()
            
            example=tf.train.Example(features=tf.train.Features(feature={
                "img" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                "label" : tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                }))
            
            writer.write(example.SerializeToString())             
            
            if i % 10000 == 0:
                print('%d pics processed' % i,"time: ", time.time()-begin)
        

if __name__ == "__main__":
    
    parser=arg_parse()
    save_dir=parser.save_dir
    read_dir=parser.read_dir
    
    begin=time.time()
    
    main()