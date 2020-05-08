import argparse
import glob
import math
import os
from collections import deque

import cv2
import numpy as np
from scipy import io as sio

from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils.sessinit import get_model_loader

from misc.utils import rm_n_mkdir

import json
import operator

import time

####

#/home/dm1/shikhar/hover_net_modified/v2_multitask/np_hv/07/model-35854.index

MODEL_DIR = '/home/dm1/shikhar/hover_net_modified/v2_multitask/np_hv/07'
#'/home/dm1/shikhar/hover_net_modified/v2_multitask_short/np_hv/07'
#'/home/dm1/shikhar/hover_net_modified/v1_multitask/np_hv/07'
#/home/dm1/shikhar/hover_net_modified/v2_multitask/np_hv/07
#/home/dm1/shikhar/hover_net_modified/v2_multitask_short/np_hv/07

def get_best_chkpts(path, metric_name, comparator='>'):
    """
    Return the best checkpoint according to some criteria.
    Note that it will only return valid path, so any checkpoint that has been
    removed wont be returned (i.e moving to next one that satisfies the criteria
    such as second best etc.)

    Args:
        path: directory contains all checkpoints, including the "stats.json" file
    """
    stat_file = path + '/stats.json'
    ops = {
            '>': operator.gt,
            '<': operator.lt,
          }

    op_func = ops[comparator]
    with open(stat_file) as f:
        info = json.load(f)
    
    if comparator == '>':
        best_value  = -float("inf")
    else:
        best_value  = +float("inf")

    best_chkpt = None
    for epoch_stat in info:
        epoch_value = epoch_stat[metric_name]
        if op_func(epoch_value, best_value):
            chkpt_path = "%s/model-%d.index" % (path, epoch_stat['global_step'])
            if os.path.isfile(chkpt_path):
                selected_stat = epoch_stat
                best_value  = epoch_value
                best_chkpt = chkpt_path
    return best_chkpt, selected_stat

####


if __name__ == '__main__':
    '''parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    args = parser.parse_args()
        
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))'''

    save_dir = MODEL_DIR
    inf_auto_metric = 'valid_acc'
    inf_auto_comparator = '>'

    print('-----Finding best checkpoint Basing On "%s" Through "%s" Comparison' % \
                (inf_auto_metric, inf_auto_comparator))
    model_path, stat = get_best_chkpts(save_dir, inf_auto_metric, inf_auto_comparator)
    print('Best checkpoint: %s' % model_path)
    print('Having Following Statistics:')
    for key, value in stat.items():
        print('\t%s: %s' % (key, value))
