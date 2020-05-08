
import glob
import os

import cv2
import numpy as np
import scipy.io as sio
from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from config import Config
from definitions import ROOT_DIR
###########################################################################
if __name__ == '__main__':
    
    #cfg = Config()

    extract_type = 'mirror' # 'valid' for fcn8 segnet etc.
                            # 'mirror' for u-net etc.
    # check the patch_extractor.py 'main' to see the different

    # orignal size (win size) - input size - output size (step size)
    # 512x512 - 256x256 - 256x256 fcn8, dcan, segnet
    # 536x536 - 268x268 - 84x84   unet, dist
    # 540x540 - 270x270 - 80x80   xy, hover
    # 504x504 - 252x252 - 252x252 micronet
    step_size = [80, 80] # should match self.train_mask_shape (config.py) 
    win_size  = [540, 540] # should be at least twice time larger than 
                           # self.train_base_shape (config.py) to reduce 
                           # the padding effect during augmentation

    xtractor = PatchExtractor(win_size, step_size)
    print(ROOT_DIR)
   
    ### ------ MODIFY THE BELOW Depending on the Dataset: -------
       
    img_ext = '.png' # (TIF: Kumar, PNG: All others)
    img_dir = ROOT_DIR + '/../data/CoNSeP/Test/Images/' # Path to Original Images
    ann_dir = ROOT_DIR + '/../data/CoNSeP/Test/Labels/' # Path to Annotations with the same name
    out_dir = ROOT_DIR + "/../data/CoNSeP/Test/%dx%d_%dx%d" % \
                        (win_size[0], win_size[1], step_size[0], step_size[1])
    nr_types = 5 # (CoNSeP,MoNuSAC: 5, All others: None)
    type_classification = True # (CoNSeP,MoNuSAC: True, All others: False)

    #### ------- ------------------------------------------ ----------

    print(img_dir)


    file_list = glob.glob('%s/*%s' % (img_dir, img_ext))
    file_list.sort() 
    print('started')
    rm_n_mkdir(out_dir)
    for filename in file_list:
        filename = os.path.basename(filename)
        basename = filename.split('.')[0]
        print(filename)
        #print(img_dir + basename + img_ext)
        img = cv2.imread(img_dir + basename + img_ext)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
        if type_classification:
            # assumes that ann is HxWx2 (nuclei class labels are available at index 1 of C) 
            ann = sio.loadmat(ann_dir + basename + '.mat')
            ann_inst = ann['inst_map']
            ann_type = ann['type_map']
            
            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 3) | (ann_type == 4)] = 3
            ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

            assert np.max(ann_type) <= nr_types-1, \
                            "Only %d types of nuclei are defined for training"\
                            "but there are %d types found in the input image." % (nr_types, np.max(ann_type)) 

            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype('int32')             
        else:
            # assumes that ann is HxW
            ann_inst = sio.loadmat(ann_dir + basename + '.mat')
            ann_inst = (ann_inst['inst_map']).astype('int32')
            ann = np.expand_dims(ann_inst, -1)
       
        img = np.concatenate([img, ann], axis=-1)
        sub_patches = xtractor.extract(img, extract_type)
        for idx, patch in enumerate(sub_patches):
            np.save("{0}/{1}_{2:03d}.npy".format(out_dir, basename, idx), patch)

print('done')