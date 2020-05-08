import tensorflow as tf
from .misc import * 
#from ..definitions import ROOT_DIR

#### Training parameters
###
# np+hv : double branches nework, 
#     1 branch nuclei pixel classification (segmentation)
#     1 branch regressing horizontal/vertical coordinate w.r.t the (supposed) 
#     nearest nuclei centroids, coordinate is normalized to 0-1 range
#
# np+dst: double branches nework
#     1 branch nuclei pixel classification (segmentation)
#     1 branch regressing nuclei instance distance map (chessboard in this case),
#     the distance map is normalized to 0-1 range

# Constants
ROOT_DIR = '/home/dm1/shikhar/hover_net_modified/src'
PRETASK_FROZEN_EPOCHS = 15
PRETASK_UNFROZEN_EPOCHS = 15
TASK_EPOCHS = 55
PRETASK_LEARNING_RATE = 1.0e-4

np_hv = {
    'train_input_shape' : [270, 270],
    'train_mask_shape'  : [ 80,  80],
    'infer_input_shape' : [270, 270],
    'infer_mask_shape'  : [ 80,  80], 

    'training_phase'    : [
        # CoNSeP
        {   # Learn End-to-End (Both Encoder/Decoder) for the Task
            # == START: Dataset specific arguments == 
           
            'train_dir' : [ROOT_DIR + '/../data/CoNSeP/Train/540x540_80x80/'],
            'valid_dir' : [ROOT_DIR + '/../data/CoNSeP/Test/540x540_80x80/'],
             # == END: Dataset specific arguments == 

            'nr_epochs': PRETASK_UNFROZEN_EPOCHS,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (PRETASK_LEARNING_RATE, [('15', 1.0e-4)]), 
            },
            # path to load, -1 to auto load checkpoint from previous phase, 
            # None to start from scratch
            'pretrained_path'  : ROOT_DIR + '/../ImageNet-ResNet50-Preact.npz',
            'train_batch_size' : 8, # unfreezing everything will
            'infer_batch_size' : 16,

            'model_flags' : {
                'freeze' : False,
                'type_classification':True,
                'nr_types': 5,
            # ! nr_types will replace nr_classes if type_classification=True
                'nr_classes': 2, # Nuclei Pixels vs Background
                'nuclei_type_dict': {
                    'other': 1, # ! Please ensure the matching ID is unique
                    'inflammatory': 2,
                    'epithelial': 3,
                    'spindle-shaped': 4,
                    }
            }
        },
        # CPM17
        { # Learn Encoder for the Task
            # == START: Dataset specific arguments == 
            'train_dir' : [ROOT_DIR + '/../data/cpm17/train/540x540_80x80/'],
            'valid_dir' : [ROOT_DIR + '/../data/cpm17/test/540x540_80x80/'],
             # == END: Dataset specific arguments == 

            'nr_epochs': PRETASK_UNFROZEN_EPOCHS,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (PRETASK_LEARNING_RATE, [('5', PRETASK_LEARNING_RATE)]), 
            },
            # path to load, -1 to auto load checkpoint from previous phase, 
            # None to start from scratch
            'pretrained_path'  : -1,
            'train_batch_size' : 8, # unfreezing everything will
            'infer_batch_size' : 16,

            'model_flags' : {
                'freeze' : False,
                'type_classification': False,
                'nr_classes': 2, # Nuclei Pixels vs Background,
                'nuclei_type_dict': {},
                'nr_types': -1
            }
        },
        # Kumar        
        { # Learn Encoder for the Task
            # == START: Dataset specific arguments == 

            'train_dir' : [ROOT_DIR + '/../data/kumar/train/540x540_80x80/'],
            'valid_dir' : [ROOT_DIR + '/../data/kumar/test_diff/540x540_80x80/'],
             # == END: Dataset specific arguments == 

            'nr_epochs': PRETASK_UNFROZEN_EPOCHS,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (PRETASK_LEARNING_RATE, [('5', PRETASK_LEARNING_RATE)]), 
            },
            # path to load, -1 to auto load checkpoint from previous phase, 
            # None to start from scratch
            'pretrained_path'  : -1,
            'train_batch_size' : 8, # unfreezing everything will
            'infer_batch_size' : 16,

            'model_flags' : {
                'freeze' : False,
                'type_classification': False,
                'nr_classes': 2, # Nuclei Pixels vs Background
                'nuclei_type_dict': {},
                'nr_types': -1
            }
        },

        # MoNuSAC 
        {   # Train decoder on MoNuSAC based on multi-task encoded representations 

            # == START: Dataset specific arguments == 
            'train_dir' : [ROOT_DIR + '/../MoNuSAC_processed/train/540x540_80x80/'],
            'valid_dir' : [ROOT_DIR + '/../MoNuSAC_processed/valid/540x540_80x80/'],
             # == END: Dataset specific arguments == 

            'nr_epochs': TASK_EPOCHS,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (1.0e-4, [('30', 1.0e-5)]), 
            },
            'pretrained_path'  : -1, #ROOT_DIR + '/../ImageNet-ResNet50-Preact.npz'
            'train_batch_size' : 8,
            'infer_batch_size' : 16,

            'model_flags' : {
                'freeze' : True,
                'type_classification': True,
                'nr_types': 5,
            # ! nr_types will replace nr_classes if type_classification=True
                'nr_classes': 2, # Nuclei Pixels vs Background
                'nuclei_type_dict': {
                    'Epithelial': 1, # ! Please ensure the matching ID is unique
                    'Lymphocyte': 2,
                    'Macrophage': 3,
                    'Neutrophil': 4,
                    }
            }
        },

        {   # Train Encoder as fine-tuning on MoNuSAC
            
            # == START: Dataset specific arguments == 
            'train_dir' : [ROOT_DIR + '/../MoNuSAC_processed/train/540x540_80x80/'],
            'valid_dir' : [ROOT_DIR + '/../MoNuSAC_processed/valid/540x540_80x80/'],
             # == END: Dataset specific arguments == 

            'nr_epochs': TASK_EPOCHS,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (1.0e-4, [('30', 1.0e-5)]), 
            },
            # path to load, -1 to auto load checkpoint from previous phase, 
            # None to start from scratch
            'pretrained_path'  : -1,
            'train_batch_size' : 4, # unfreezing everything will
            'infer_batch_size' : 16,

            'model_flags' : {
                'freeze' : False,
                'type_classification': True,
                'nr_types': 5,
            # ! nr_types will replace nr_classes if type_classification=True
                'nr_classes': 2, # Nuclei Pixels vs Background
                'nuclei_type_dict': {
                    'Epithelial': 1, # ! Please ensure the matching ID is unique
                    'Lymphocyte': 2,
                    'Macrophage': 3,
                    'Neutrophil': 4,
                    }
            }
        }
    ],

    'loss_term' : {'bce' : 1, 'dice' : 1, 'mse' : 2, 'msge' : 1}, 

    'optimizer'           : tf.train.AdamOptimizer,

    'inf_auto_metric'   : 'valid_dice',
    'inf_auto_comparator' : '>',
    'inf_batch_size' : 16,
}

# Secondary Network [NOT USED]
np_dist = {
    'train_input_shape' : [270, 270],
    'train_mask_shape'  : [ 80,  80],
    'infer_input_shape' : [270, 270],
    'infer_mask_shape'  : [ 80,  80], 

    'training_phase'    : [
        {
            'nr_epochs': 50,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (1.0e-4, [('25', 1.0e-5)]), 
            },
            'pretrained_path'  : ROOT_DIR + '/ImageNet-ResNet50-Preact.npz',
            'train_batch_size' : 8,
            'infer_batch_size' : 16,

            'model_flags' : {
                'freeze' : True
            }
        },

        {
            'nr_epochs': 50,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (1.0e-4, [('25', 1.0e-5)]), 
            },
            # path to load, -1 to auto load checkpoint from previous phase, 
            # None to start from scratch
            'pretrained_path'  : -1,
            'train_batch_size' : 4, # unfreezing everything will
            'infer_batch_size' : 16,

            'model_flags' : {
                'freeze' : False
            }
        }
    ],
  
    'optimizer'         : tf.train.AdamOptimizer,

    'inf_auto_metric'   : 'valid_dice',
    'inf_auto_comparator' : '>',
    'inf_batch_size' : 16,
}
