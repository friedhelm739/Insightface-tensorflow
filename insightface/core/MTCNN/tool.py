# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
import numpy as np
import cv2
import tensorflow as tf

def IoU(box, boxes):
    """
    Compute IoU between detect box and face boxes

    Parameters:
    ----------
    box: numpy array , shape (4, ): x1, y1, x2, y2
         random produced box
    boxes: numpy array, shape (n, 4): x1, y1, w, h
         input ground truth face boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """   
    
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = boxes[:, 2]*boxes[:, 3]
    
    x_right=boxes[:, 2]+boxes[:, 0]
    y_bottom=boxes[:, 3]+boxes[:, 1]
    
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], x_right)
    yy2 = np.minimum(box[3], y_bottom)

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


def NMS(box,_overlap,mode="default"):
    
    if len(box) == 0:
        return []

    pick = []
    x_min = box[:,0]
    y_min = box[:,1]
    x_max = box[:,2]
    y_max = box[:,3]
    score = box[:,4]

    area = (x_max-x_min)*(y_max-y_min)
    idxs = score.argsort()[::-1]

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x_min[i],x_min[idxs[1:]])
        yy1 = np.maximum(y_min[i],y_min[idxs[1:]])
        xx2 = np.minimum(x_max[i],x_max[idxs[1:]])
        yy2 = np.minimum(y_max[i],y_max[idxs[1:]])

        w = np.maximum(xx2-xx1,0)
        h = np.maximum(yy2-yy1,0)

        square=w*h
        
        if(mode != "Minimum"):
            overlap = square / (area[idxs[1:]] + area[i] - square)
        else:
            overlap = square / np.minimum(area[idxs[1:]] , area[i])
            
        idxs = idxs[np.where(overlap < _overlap)[0]+1]
    
    return pick


def flip(img,facemark):
    img=cv2.flip(img,1)
    facemark=np.asarray([(1-x,y) for (x,y) in facemark])
    facemark[[0,1]]=facemark[[1,0]]
    facemark[[3,4]]=facemark[[4,3]]   
    return (img,facemark)


def read_single_tfrecord(addr,_batch_size,shape):
    
    filename_queue = tf.train.string_input_producer([addr],shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 

    features = tf.parse_single_example(serialized_example,
                                   features={
                                   'img':tf.FixedLenFeature([],tf.string),
                                   'label':tf.FixedLenFeature([],tf.int64),                                   
                                   'roi':tf.FixedLenFeature([4],tf.float32),
                                   'landmark':tf.FixedLenFeature([10],tf.float32),
                                   })
    img=tf.decode_raw(features['img'],tf.uint8)
    label=tf.cast(features['label'],tf.int32)
    roi=tf.cast(features['roi'],tf.float32)
    landmark=tf.cast(features['landmark'],tf.float32)
    img = tf.reshape(img, [shape,shape,3])     
    img=(tf.cast(img,tf.float32)-127.5)/128 
    min_after_dequeue = 10000
    batch_size = _batch_size
    capacity = min_after_dequeue + 10 * batch_size
    image_batch, label_batch, roi_batch, landmark_batch = tf.train.shuffle_batch([img,label,roi,landmark], 
                                                        batch_size=batch_size, 
                                                        capacity=capacity, 
                                                        min_after_dequeue=min_after_dequeue,
                                                        num_threads=7)  
    
    label_batch = tf.reshape(label_batch, [batch_size])
    roi_batch = tf.reshape(roi_batch,[batch_size,4])
    landmark_batch = tf.reshape(landmark_batch,[batch_size,10])
    
    return image_batch, label_batch, roi_batch, landmark_batch

   
def read_multi_tfrecords(addr,_batch_size,shape):
    
    pos_dir,part_dir,neg_dir,landmark_dir = addr
    pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size = _batch_size   
    
    pos_image,pos_label,pos_roi,pos_landmark = read_single_tfrecord(pos_dir, pos_batch_size, shape)
    part_image,part_label,part_roi,part_landmark = read_single_tfrecord(part_dir, part_batch_size, shape)
    neg_image,neg_label,neg_roi,neg_landmark = read_single_tfrecord(neg_dir, neg_batch_size, shape)
    landmark_image,landmark_label,landmark_roi,landmark_landmark = read_single_tfrecord(landmark_dir, landmark_batch_size, shape)

    images = tf.concat([pos_image,part_image,neg_image,landmark_image], 0, name="concat/image")
    labels = tf.concat([pos_label,part_label,neg_label,landmark_label],0,name="concat/label")
    rois = tf.concat([pos_roi,part_roi,neg_roi,landmark_roi],0,name="concat/roi")
    landmarks = tf.concat([pos_landmark,part_landmark,neg_landmark,landmark_landmark],0,name="concat/landmark")
    
    return images,labels,rois,landmarks    
    

def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs,max_delta= 0.2)
    inputs = tf.image.random_saturation(inputs,lower = 0.5, upper= 1.5)

    return inputs
