# -*- coding: utf-8 -*-
"""

@author: friedhelm
"""
import sys
sys.path.append("../")

import numpy as np
from core import Arcface_model,config
from .evaluate_tool import run_emdb,calculate_roc
import io
from scipy import misc
import cv2
import pickle
import tensorflow as tf

def load_bin(path, image_size=112):
    
    print('reading %s' % path)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')

    num = len(bins)
    images = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)
    images_f = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)

    cnt = 0
    for bin in bins:
        img = misc.imread(io.BytesIO(bin)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img,(image_size, image_size))
        img_f = np.fliplr(img)
        img = (img - 127.5)/128
        img_f = (img_f - 127.5)/128
        images[cnt] = img
        images_f[cnt] = img_f
        cnt += 1
    print('done!')
    return (images, images_f, issame_list)


def evaluation(sess, images, images_f, issame_list, eval_batch_size, image_size, **params):

    issame_list = np.array(issame_list)
    embds_arr = run_emdb(sess, images, image_size, eval_batch_size, **params)
    embds_f_arr = run_emdb(sess,images_f, image_size, eval_batch_size, **params)
    embeddings = embds_arr/np.linalg.norm(embds_arr, axis=1, keepdims=True) + embds_f_arr/np.linalg.norm(embds_f_arr, axis=1, keepdims=True)
    
    embeddings_1 = embeddings[0::2]
    embeddings_2 = embeddings[1::2]    
    
    thresholds = np.arange(0, 4, 0.01)
    
    tpr, fpr, accuracy, best_thresholds = calculate_roc(embeddings_1, embeddings_2, issame_list, thresholds)
    
    accuracy = np.mean(accuracy)
    
    return tpr, fpr, accuracy, best_thresholds


if __name__ == "__main__":

    img_size=112
    batch_size=256
    model_path="../model/Arcface-300000"
    
    
    image=tf.placeholder(tf.float32,[batch_size,img_size,img_size,3],name='image')
    train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_dropout')
    train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_bn')    
    
    embd, _ = Arcface_model.get_embd(image, train_phase_dropout, train_phase_bn, config.model_params)
    saver=tf.train.Saver()

    tf_config = tf.ConfigProto(allow_soft_placement=True)  
    tf_config.gpu_options.allow_growth = True 
    with tf.Session(config = tf_config) as sess:
        saver.restore(sess,model_path)
        for dataset_path in config.eval_datasets:
            tpr, fpr, accuracy, best_thresholds = evaluation(sess, batch_size, img_size, dataset_path, dropout_flag=config.eval_dropout_flag, bn_flag=config.eval_bn_flag, embd=embd, image=image, train_phase_dropout=train_phase_dropout, train_phase_bn=train_phase_bn) 
            print("%s datasets get %.3f acc,best_thresholds are: "%(dataset_path.split("/")[-1].split(".")[0],accuracy), best_thresholds)