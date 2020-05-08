import argparse
from glob import glob
import math
import os
from collections import deque
import json
import operator
import shutil
import time
import cv2
import numpy as np
from scipy import io as sio
from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils.sessinit import get_model_loader
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (binary_dilation, binary_fill_holes,
                                      distance_transform_cdt,
                                      distance_transform_edt)
from skimage.morphology import remove_small_objects, watershed

import postproc.hover
import postproc.dist
import postproc.other

from misc.viz_utils import visualize_instances
from misc.utils import get_inst_centroid
from metrics.stats_utils import remap_label

from config import Config


def rm_n_mkdir(dir_path):
    if (os.path.isdir(dir_path)):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

energy_mode = 2
marker_mode = 2 

class_id_mapping = {
            1: 'Epithelial', # ! Please ensure the matching ID is unique
            2: 'Lymphocyte',
            3: 'Macrophage',
            4: 'Neutrophil',
        }




class Submission(Config):

    def __gen_prediction(self, x, predictor):

        
        """
        Using 'predictor' to generate the prediction of image 'x'

        Args:
            x : input image to be segmented. It will be split into patches
                to run the prediction upon before being assembled back            
        """    
        step_size = self.infer_mask_shape
        msk_size = self.infer_mask_shape
        win_size = self.infer_input_shape

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)
        
        im_h = x.shape[0] 
        im_w = x.shape[1]

        last_h, nr_step_h = get_last_steps(im_h, msk_size[0], step_size[0])
        last_w, nr_step_w = get_last_steps(im_w, msk_size[1], step_size[1])

        diff_h = win_size[0] - step_size[0]
        padt = diff_h // 2
        padb = last_h + win_size[0] - im_h

        diff_w = win_size[1] - step_size[1]
        padl = diff_w // 2
        padr = last_w + win_size[1] - im_w

        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), 'reflect')

        #### TODO: optimize this
        sub_patches = []
        # generating subpatches from orginal
        for row in range(0, last_h, step_size[0]):
            for col in range (0, last_w, step_size[1]):
                win = x[row:row+win_size[0], 
                        col:col+win_size[1]]
                sub_patches.append(win)

        pred_map = deque()
        while len(sub_patches) > self.inf_batch_size:
            mini_batch  = sub_patches[:self.inf_batch_size]
            sub_patches = sub_patches[self.inf_batch_size:]
            mini_output = predictor(mini_batch)[0]
            mini_output = np.split(mini_output, self.inf_batch_size, axis=0)
            pred_map.extend(mini_output)
        if len(sub_patches) != 0:
            mini_output = predictor(sub_patches)[0]
            mini_output = np.split(mini_output, len(sub_patches), axis=0)
            pred_map.extend(mini_output)

        #### Assemble back into full image
        output_patch_shape = np.squeeze(pred_map[0]).shape
        ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

        #### Assemble back into full image
        pred_map = np.squeeze(np.array(pred_map))
        pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
        pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
                        np.transpose(pred_map, [0, 2, 1, 3])
        pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1], 
                                         pred_map.shape[2] * pred_map.shape[3], ch))
        pred_map = np.squeeze(pred_map[:im_h,:im_w]) # just crop back to original size

        return pred_map

    ####
    def run(self, data_dir, output_dir, model_path, img_ext = '.png'):
        
        if(not data_dir):
            print('Using Config file path for data_dir.')
            data_dir = self.inf_data_dir
        if(not output_dir):
            print('Using Config file path for output_dir.')
            output_dir = self.inf_output_dir
        if(not model_path):
            print('Using placeholder path for model_dir.')
            model_path = '/home/dm1/shikhar/hover_net_modified/v2_multitask/np_hv/07/model-35854.index'
        if(not img_ext):
            print('Using Config img ext value img_ext.')
            img_ext = self.inf_imgs_ext

        model_constructor = self.get_model()
        pred_config = PredictConfig(
            model        = model_constructor(),
            session_init = get_model_loader(model_path),
            input_names  = self.eval_inf_input_tensor_names,
            output_names = self.eval_inf_output_tensor_names)
        predictor = OfflinePredictor(pred_config)

        #file_list = glob.glob('%s/*%s' % (data_dir, img_ext))
        #file_list.sort() # ensure same order
        #if(not file_list):
        # print('No Images found in data_dir! Check script arg-paths') 
        # Create Output Directory
        #rm_n_mkdir(output_dir)       
        # Expecting MoNuSAC's input data directory tree (Patient Name -> Image Name -> )
        
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        os.chdir(output_dir)
        patients = [x[0] for x in os.walk(data_dir)]#Total patients in the data_path
        print(len(patients))

        for patient_loc in patients:
            patient_name = patient_loc[len(data_dir)+1:]#Patient name
            print(patient_name, flush=True)
            
            ## To make patient's name directory in the destination folder
            try:
                os.mkdir(patient_name)
            except OSError:
                print ("\n Creation of the patient's directory %s failed" % patient_name,  flush=True)

            sub_images = glob(str(patient_loc) + '/*' + str(img_ext))
            for sub_image_loc in sub_images:
                sub_image_name = sub_image_loc[len(data_dir)+len(patient_name)+1:-4]        
                print(sub_image_name)
                
                ## To make sub_image directory under the patient's folder
                sub_image = './'+patient_name + sub_image_name #Destination path
                try:
                    os.mkdir(sub_image)
                except OSError:
                    print ("\n Creation of the patient's directory %s failed" % sub_image)
                
                image_name = sub_image_loc
                if(img_ext == '.svs'):
                    img = openslide.OpenSlide(image_name)
                    cv2.imwrite(sub_image_loc[:-4]+'.png', np.array(img.read_region((0,0),0,img.level_dimensions[0])))      
                    img = cv2.imread(sub_image_loc[:-4]+'.png')
                else:
                    img = cv2.imread(image_name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                ## Generate Prediction Map
                pred_map = self.__gen_prediction(img, predictor)
                pred = pred_map
                # Process Prediction Map
                pred_inst = pred[...,self.nr_types:]
                pred_type = pred[...,:self.nr_types]
                pred_inst = np.squeeze(pred_inst)
                pred_type = np.argmax(pred_type, axis=-1)
                pred_inst = postproc.hover.proc_np_hv(pred_inst, 
                            marker_mode=marker_mode,
                            energy_mode=energy_mode, rgb=img)
                pred_inst = remap_label(pred_inst, by_size=True)
                
                # Map Instances to Labels for creating submission format
                pred_id_list = list(np.unique(pred_inst))[1:] # exclude background ID
                pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
                for idx, inst_id in enumerate(pred_id_list):
                    inst_type = pred_type[pred_inst == inst_id]
                    type_list, type_pixels = np.unique(inst_type, return_counts=True)
                    type_list = list(zip(type_list, type_pixels))
                    type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
                    inst_type = type_list[0][0]
                    if inst_type == 0: # ! pick the 2nd most dominant if exist
                        if len(type_list) > 1:
                            inst_type = type_list[1][0]
                        else:
                            print('[Warn] Instance has `background` type' )
                    pred_inst_type[idx] = inst_type
                
                # Write Instance Maps based on their Classes/Labels to the folders
                for class_id in range(1,self.nr_types):
                    separated_inst = pred_inst.copy()
                    separated_inst[pred_inst_type[separated_inst-1]!=[class_id]] = 0
                    # Create directory for each label
                    label = class_id_mapping[class_id]
                    sub_path = sub_image+'/'+label
                    try:
                        os.mkdir(sub_path)
                    except OSError:
                        print ("Creation of the directory %s failed" % label)
                    else:
                        print ("Successfully created the directory %s " % label)

                    sio.savemat(sub_path +'/maskorempty.mat', 
                        {'n_ary_mask'  :  separated_inst})
            

####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='data_dir path for test image data')
    parser.add_argument('--output_dir', help='path for model predictions')
    parser.add_argument('--img_ext', help='extension for test image. Default: .png')
    parser.add_argument('--model_path', help='path to model being tested (.index file)')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')


    args = parser.parse_args()
        
    if args.data_dir:
        data_dir = args.data_dir
    if args.output_dir:
        output_dir = args.output_dir
    if args.output_dir:
        img_ext = args.img_ext
    if args.model_path:
        model_path = args.model_path
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        n_gpus = len(args.gpu.split(','))

    submission = Submission()
    submission.run(output_dir=output_dir, data_dir=data_dir, model_path = model_path, img_ext = img_ext)    
