# -*- coding: utf-8 -*-
"""

@author: friedhelm
"""
import sys
sys.path.append("../")

import numpy as np
from core import config
import math
from sklearn.model_selection import KFold


def run_emdb(sess,eval_images,image_size,eval_batch_size, **params):
    """
    used for run out embeddings

    Parameters:
    ----------
        sess : tf.Session()
        eval_images : numpy.array, shape(n,image_size,image_size,3 )
        image_size : int
        eval_batch_size : int
        dropout_flag : bool
        bn_flag : bool
        embd : tensor, shape(n, config.model_params["embd_size"])
        image : tensor, tf.placeholder() 
        train_phase_dropout : tensor, tf.placeholder()       
        train_phase_bn : tensor, tf.placeholder()          
        
    Returns:
    -------
        numpy.array 
    
        run_emdb
    """         
    dropout_flag = params["dropout_flag"]
    bn_flag = params["bn_flag"]
    embd = params["embd"]
    image = params["image"]
    train_phase_dropout = params["train_phase_dropout"]
    train_phase_bn = params["train_phase_bn"]    

    batch_num = len(eval_images)//eval_batch_size
    left = int(len(eval_images)/eval_batch_size)
    embd_list=[]
    
    for i in range(batch_num):
        image_batch=eval_images[i*eval_batch_size:(i+1)*eval_batch_size]
        eval_embd=sess.run(embd,feed_dict={image:image_batch,train_phase_dropout:dropout_flag,train_phase_bn:bn_flag})
        embd_list+=list(eval_embd)
        
    if left:
        image_batch = np.zeros([eval_batch_size, image_size, image_size, 3])
        image_batch[:left, :, :, :] = eval_images[-left:]
        eval_embd=sess.run(embd,feed_dict={image:image_batch,train_phase_dropout:dropout_flag,train_phase_bn:bn_flag})
        embd_list+=list(eval_embd)        
        
    return np.array(embd_list)


def calculate_distance(embeddings_1, embeddings_2, dist_sign="Euclidian"):
    """
    used for calculate distance

    Parameters:
    ----------
        embeddings_1 : numpy.array, shape(n, config.model_params["embd_size"])
        embeddings_2 : numpy.array, shape(n, config.model_params["embd_size"])
        dist_sign : str          
        
    Returns:
    -------
        numpy.array 
    
        calculate_distance
    """      
    if(dist_sign=="Euclidian"):
        # Euclidian distance ,after nomalization , dist(eula) = 2 * (1 - dis(cos) )          
        embeddings_1 = embeddings_1/np.linalg.norm(embeddings_1, axis=1, keepdims=True)
        embeddings_2 = embeddings_2/np.linalg.norm(embeddings_2, axis=1, keepdims=True)
        diff = np.subtract(embeddings_1, embeddings_2)
        dist = np.sum(np.square(diff),1)
    else :
        # Distance based on cosine similarity        
        dot = np.sum(np.multiply(embeddings_1, embeddings_2), axis=1)
        norm = np.linalg.norm(embeddings_1, axis=1) * np.linalg.norm(embeddings_2, axis=1)
        similarity = dot/norm
        dist = np.arccos(similarity) / math.pi        
        
    return dist   


def calculate_roc(embeddings_1, embeddings_2, eval_labels, thresholds, nrof_folds=10, dist_sign="Euclidian"):
    """
    used for calculate ROC

    Parameters:
    ----------
        embeddings_1 : numpy.array, shape(n, config.model_params["embd_size"])
        embeddings_2 : numpy.array, shape(n, config.model_params["embd_size"])
        eval_labels : list, shape(k, )        
        thresholds : numpy.array, shape(m, )        
        nrof_folds : int       
        dist_sign : str          
        
    Returns:
    -------
        int, int, numpy.array, numpy.array
    
        calculate_roc
    """     
    assert(embeddings_1.size == embeddings_2.size)
    thresholds_len = len(thresholds)
    indices = np.arange(min(len(eval_labels), embeddings_1.shape[0]))
    kfold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds,thresholds_len))
    fprs = np.zeros((nrof_folds,thresholds_len))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    dist=calculate_distance(embeddings_1,embeddings_2,dist_sign)
    
    for kfold_idx, (train_data,test_data) in enumerate(kfold.split(indices)):
        
        train_acc = np.zeros((thresholds_len))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, train_acc[threshold_idx] = calculate_acc(threshold, dist[train_data], eval_labels[train_data])
            tprs[kfold_idx,threshold_idx], fprs[kfold_idx,threshold_idx], _ = calculate_acc(threshold, dist[test_data], eval_labels[test_data])
            
        best_threshold = np.argmax(train_acc)
        best_thresholds[kfold_idx] = best_threshold*0.01
        _, _, accuracy[kfold_idx] = calculate_acc(thresholds[best_threshold], dist[test_data], eval_labels[test_data])
    
    tpr=np.mean(tprs, 0)
    fpr=np.mean(fprs, 0)    
    
    return tpr, fpr, accuracy, best_thresholds



def calculate_acc(threshold, dist, actual_issame):
    """
    used for run out embeddings

    Parameters:
    ----------
        tpr : int
        fpr : int     
        accuracy : int           
        
    Returns:
    -------
        numpy.array 
    
        calculate_acc
    """       
    predict_issame = np.less(dist, threshold)
    
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    
    return tpr, fpr, acc 
