# -*- coding: utf-8 -*-
"""

@author: friedhelm
"""
import os


# prepare_data parameters
mxdata_dir = "./data"
tfrecord_dir = "./data/CASIA.tfrecords"
lfw_dir = './data/lfw'
lfw_save_dir = './data/lfw_face'
eval_dir = './data/lfw_face.db'
eval_datasets_self = ['./data/lfw_face.db']
eval_datasets = ["./data/lfw.bin","./data/agedb_30.bin","./data/calfw.bin","./data/cfp_ff.bin","./data/cfp_fp.bin","./data/cplfw.bin",'./data/lfw_face.db']


# model parameters
model_params = {"backbone_type": "resnet_v2_m_50",
                "out_type" : "E",
                "bn_decay": 0.9,
                "weight_decay": 0.0005,
                "keep_prob":0.4,
                "embd_size":512}


# training parameters
s = 64.0
m = 0.5
class_num = 85742
lr_steps = [40000, 60000, 80000]
lr_values = [0.004, 0.002, 0.0012, 0.0004]
#lr_steps = [200000, 280000, 320000]
#lr_values = [0.1, 0.001, 0.0001, 0.00001]
momentum = 0.9
addrt="./data/CASIA.tfrecords"
model_patht="./model/Arcface_model"
img_size = 112
batch_size = 128
addr="../data/CASIA.tfrecords"
model_name="Arcface"
train_step=1000001
model_path="../model/Arcface_model"
gpu_num=2
model_save_gap = 30000


# evaluation parameters
eval_dropout_flag = False
eval_bn_flag = False


# face database parameters
custom_dir = '../data/custom'
arc_model_name = 'Arcface-330000'
arc_model_path = './model/Arcface_model/Arcface-330000'

base_dir = './model/MTCNN_model'
mtcnn_model_path = [os.path.join(base_dir,"Pnet_model/Pnet_model.ckpt-20000"),
                    os.path.join(base_dir,"Rnet_model/Rnet_model.ckpt-40000"),
                    os.path.join(base_dir,"Onet_model/Onet_model.ckpt-40000")] 
embds_save_dir = "../data/face_db"
