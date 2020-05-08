import importlib
import random

import cv2
import numpy as np
import tensorflow as tf
from tensorpack import imgaug

from loader.augs import (BinarizeLabel, GaussianBlur, GenInstanceDistance,
                         GenInstanceHV, MedianBlur, GenInstanceUnetMap,
                         GenInstanceContourMap)
from definitions import ROOT_DIR

class Config(object):
    def __init__(self, ):
        
        # Defined static settings for all tasks 
        self.seed = 10 
        mode = 'hover'
        self.model_type = 'np_hv'

        #### Dynamically setting the config file into variable
        if mode == 'hover':
            config_file = importlib.import_module('opt.hover_multitask') # np_hv, np_dist
            print('(Config) Using Multi-Task Settings..')
        else:
            config_file = importlib.import_module('opt.other') # fcn8, dcan, etc.

        config_dict = config_file.__getattribute__(self.model_type)

        for variable, value in config_dict.items():
            self.__setattr__(variable, value)
        #### Training data

        # patches are stored as numpy arrays with N channels
        # ordering as [Image][Nuclei Pixels][Nuclei Type][Additional Map]
        # Ex: with type_classification=True
        #     HoVer-Net: RGB - Nuclei Pixels - Type Map - Horizontal and Vertical Map
        # Ex: with type_classification=False
        #     Dist     : RGB - Nuclei Pixels - Distance Map
        data_code_dict = {
            'unet'     : '536x536_84x84',
            'dist'     : '536x536_84x84',
            'fcn8'     : '512x512_256x256',
            'dcan'     : '512x512_256x256',
            'segnet'   : '512x512_256x256',
            'micronet' : '504x504_252x252', 
            'np_hv'    : '540x540_80x80',
            'np_dist'  : '540x540_80x80',
        }

        self.data_ext = '.npy' 
        # list of directories containing validation patches. 
        # For both train and valid directories, a comma separated list of directories can be used

        # number of processes for parallel processing input
        self.nr_procs_train = 8 
        self.nr_procs_valid = 4

        self.input_norm  = True # normalize RGB to 0-1 range
        ####
        # v1_multitask, v2_multitask, v2_multitask_large_batch_high_learn, 
        exp_id = 'v2_multitask_slow'
        model_id = '%s' % self.model_type
        self.model_name = '%s/%s' % (exp_id, model_id)
        # loading chkpts in tensorflow, the path must not contain extra '/'
        self.log_path = ROOT_DIR +  '/../' # log root path - modify according to needs
        self.save_dir = '%s/%s' % (self.log_path, self.model_name) # log file destination

        #### Info for running inference
        self.inf_auto_find_chkpt = True 
        # path to checkpoints will be used for inference, replace accordingly
        self.inf_model_path  = self.save_dir + '/model-19640.index'

        # output will have channel ordering as [Nuclei Type][Nuclei Pixels][Additional]
        # where [Nuclei Type] will be used for getting the type of each instance
        # while [Nuclei Pixels][Additional] will be used for extracting instances

        self.inf_imgs_ext = '.png'
        self.inf_data_dir = ROOT_DIR + '/../MoNuSAC_processed/Valid_Images/' 
        self.inf_output_dir = ROOT_DIR + '/../MoNuSAC_processed/Overlay/%s/%s' % (exp_id, model_id)

        # for inference during evalutaion mode i.e run by infer.py
        self.eval_inf_input_tensor_names = ['images']
        self.eval_inf_output_tensor_names = ['predmap-coded']
        # for inference during training mode i.e run by trainer.py
        self.train_inf_output_tensor_names = ['predmap-coded', 'truemap-coded']

    def get_model(self):
        if self.model_type == 'np_hv':
            model_constructor = importlib.import_module('model.graph_multitask')
            model_constructor = model_constructor.Model_NP_HV
        elif self.model_type == 'np_dist':
            model_constructor = importlib.import_module('model.graph')
            model_constructor = model_constructor.Model_NP_DIST 
        else:
            model_constructor = importlib.import_module('model.%s' % self.model_type)
            model_constructor = model_constructor.Graph       
        return model_constructor # NOTE return alias, not object

    # refer to https://tensorpack.readthedocs.io/modules/dataflow.imgaug.html for 
    # information on how to modify the augmentation parameters
    def get_train_augmentors(self, input_shape, output_shape, type_classification, view=False):
        
        print('GET_TRAIN_AUGMENTORS Parameters: ',input_shape, output_shape, type_classification)
        shape_augs = [
            imgaug.Affine(
                        shear=5, # in degree
                        scale=(0.8, 1.2),
                        rotate_max_deg=179,
                        translate_frac=(0.01, 0.01),
                        interp=cv2.INTER_NEAREST,
                        border=cv2.BORDER_CONSTANT),
            imgaug.Flip(vert=True),
            imgaug.Flip(horiz=True),
            imgaug.CenterCrop(input_shape),
        ]

        input_augs = [
            imgaug.RandomApplyAug(
                imgaug.RandomChooseAug(
                    [
                    GaussianBlur(),
                    MedianBlur(),
                    imgaug.GaussianNoise(),
                    ]
                ), 0.5),
            # standard color augmentation
            imgaug.RandomOrderAug(
                [imgaug.Hue((-8, 8), rgb=True), 
                imgaug.Saturation(0.2, rgb=True),
                imgaug.Brightness(26, clip=True),  
                imgaug.Contrast((0.75, 1.25), clip=True),
                ]),
            imgaug.ToUint8(),
        ]

        label_augs = []
        if self.model_type == 'unet' or self.model_type == 'micronet':
            label_augs =[GenInstanceUnetMap(crop_shape=output_shape)]
        if self.model_type == 'dcan':
            label_augs =[GenInstanceContourMap(crop_shape=output_shape)]
        if self.model_type == 'dist':
            label_augs = [GenInstanceDistance(crop_shape=output_shape, inst_norm=False)]
        if self.model_type == 'np_hv':
            label_augs = [GenInstanceHV(crop_shape=output_shape)]
        if self.model_type == 'np_dist':
            label_augs = [GenInstanceDistance(crop_shape=output_shape, inst_norm=True)]

        if not type_classification:            
            label_augs.append(BinarizeLabel())

        if not view:
            label_augs.append(imgaug.CenterCrop(output_shape))      

        return shape_augs, input_augs, label_augs

    def get_valid_augmentors(self, input_shape, output_shape, view=False):
        print(input_shape, output_shape)
        shape_augs = [
            imgaug.CenterCrop(input_shape),
        ]

        input_augs = None

        label_augs = []
        if self.model_type == 'unet' or self.model_type == 'micronet':
            label_augs =[GenInstanceUnetMap(crop_shape=output_shape)]
        if self.model_type == 'dcan':
            label_augs =[GenInstanceContourMap(crop_shape=output_shape)]
        if self.model_type == 'dist':
            label_augs = [GenInstanceDistance(crop_shape=output_shape, inst_norm=False)]
        if self.model_type == 'np_hv':
            label_augs = [GenInstanceHV(crop_shape=output_shape)]
        if self.model_type == 'np_dist':
            label_augs = [GenInstanceDistance(crop_shape=output_shape, inst_norm=True)]
        label_augs.append(BinarizeLabel())

        if not view:
            label_augs.append(imgaug.CenterCrop(output_shape))        

        return shape_augs, input_augs, label_augs
