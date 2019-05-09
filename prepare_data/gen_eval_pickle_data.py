# -*- coding: utf-8 -*-
"""

@author: friedhelm
"""
import argparse
import pickle
import os
import numpy as np
from collections import namedtuple
from core import config

def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            print('not exists', path0, path1)
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list



def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)
 
    
def arg_parse():
    
    parser = argparse.ArgumentParser(description='Package LFW images')
    parser.add_argument('--input_dir', default=config.mxdata_dir, help='path to load')
    parser.add_argument('--output_dir', default=config.eval_dir, help='path to save.')
    
    return parser


if __name__=="__main__":
    
    args = arg_parse()
#     User = namedtuple('User', ['input_dir', 'output_dir'])
#     args = User(input_dir='./data', output_dir='./data/lfw_face.db')    
    
    lfw_dir = args.input_dir
    lfw_pairs = read_pairs(os.path.join(lfw_dir, 'pairs.txt'))

    lfw_dir = os.path.join(lfw_dir, 'lfw_face')
    lfw_paths, issame_list = get_paths(lfw_dir, lfw_pairs, 'jpg')
    
    lfw_bins = []
    i = 0
    for path in lfw_paths:
        with open(path, 'rb') as fin:
            _bin = fin.read()
            lfw_bins.append(_bin)
            i+=1
            if i%1000==0:
                print('loading lfw', i)
    
    with open(args.output_dir, 'wb') as f:
        pickle.dump((lfw_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
