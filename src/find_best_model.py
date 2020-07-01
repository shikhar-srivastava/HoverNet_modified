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
# v1.1
    [ 0.81694  0.74608  0.80668  0.61129  0.64900  0.64629]
        metrics[0].append(get_dice_1(true, pred))
        metrics[1].append(pq_info[0]) # dq
        metrics[2].append(pq_info[1]) # sq
        metrics[3].append(pq_info[2]) # pq
        metrics[4].append(get_fast_aji_plus(true, pred))
        metrics[5].append(get_fast_aji(true, pred))

# v1_multitask
# REPORTING THIS
#Performance  '/home/dm1/shikhar/hover_net_modified/src/..//v1_multitask/np_hv/07/model-33198': [ 0.79046  0.68914  0.76172  0.56644  0.60720  0.60457]

# v2_multitask
# Let's Try: /home/dm1/shikhar/hover_net_modified/src/..//v2_multitask/np_hv/07/model-42118
'''[0507 23:46:38 @monitor.py:459] DataParallelInferenceRunner/QueueInput/queue_size: 49.873
[0507 23:46:38 @monitor.py:459] QueueInput/queue_size: 48.004
[0507 23:46:38 @monitor.py:459] learning_rate: 1e-05
[0507 23:46:38 @monitor.py:459] loss-bce: 0.14295
[0507 23:46:38 @monitor.py:459] loss-dice: 0.22765
[0507 23:46:38 @monitor.py:459] loss-dice-class: 2.4458
[0507 23:46:38 @monitor.py:459] loss-mse: 0.018726
[0507 23:46:38 @monitor.py:459] loss-msge: 0.22298
[0507 23:46:38 @monitor.py:459] loss-xentropy-class: 0.17471
[0507 23:46:38 @monitor.py:459] overall-loss: 3.2515
[0507 23:46:38 @monitor.py:459] valid_acc: 0.93249
[0507 23:46:38 @monitor.py:459] valid_dice: 0.79559
[0507 23:46:38 @monitor.py:459] valid_dice_Epithelial: 0.83496
[0507 23:46:38 @monitor.py:459] valid_dice_Lymphocyte: 0.77113
[0507 23:46:38 @monitor.py:459] valid_dice_Macrophage: 0.7902
[0507 23:46:38 @monitor.py:459] valid_dice_Neutrophil: 0.83246
[0507 23:46:38 @monitor.py:459] valid_mse: 0.039461
[ 0.76476  0.63755  0.75260  0.51692  0.58312  0.57970]
        metrics[0].append(get_dice_1(true, pred))
        metrics[1].append(pq_info[0]) # dq
        metrics[2].append(pq_info[1]) # sq
        metrics[3].append(pq_info[2]) # pq
        metrics[4].append(get_fast_aji_plus(true, pred))
        metrics[5].append(get_fast_aji(true, pred))
'''

#MODEL_DIR = '/home/dm1/shikhar/hover_net_modified/v2_multitask/np_hv/07'

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
