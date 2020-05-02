"""
Process Whole slide images and their respective annotations into .mat file as per the paper & related code from  "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images"

----------------------------------------------------------------------------------------------------
Each ground truth file is stored as a .mat file, with the keys:
'inst_map'
'type_map'
'inst_type'
'inst_centroid'
 
'inst_map' is a 1000x1000 array containing a unique integer for each individual nucleus. i.e the map ranges from 0 to N, where 0 is the background and N is the number of nuclei.

'type_map' is a 1000x1000 array where each pixel value denotes the class of that pixel. The map ranges from 0 to 7, where 7 is the total number of classes in CoNSeP.

"""

import os
import openslide
from xml.dom import minidom
import numpy as np
import openslide
from openslide import open_slide  
from glob import glob
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image
import scipy
import scipy.ndimage
from shapely.geometry import Polygon
from skimage import draw
import xml.etree.ElementTree as ET
import argparse
from misc.viz_utils import visualize_instances

def main(input_loc, output_loc):

    # Read svs files from the desired path
    count = 0
    if(input_loc):
        data_path = input_loc
    else:
        data_path = '/usr/local/opt/work/hover_net_modified/MoNuSAC' #Path to read data from
    if(output_loc):
        destination_path = output_loc # Path to save n-ary masks corresponding to xml files
    else:
        destination_path = '/usr/local/opt/work/hover_net_modified/'

    os.chdir(destination_path)

    # Create MoNuSAC folder
    try:
        os.mkdir(destination_path+'/MoNuSAC_processed')
    except OSError:
        print ("Creation of the mask directory %s failed" % destination_path)
    
    # Create sub-folders in the same pattern as CoNSeP 
    # -- Images
    # -- Labels
    # -- Overlay
    try:
        os.mkdir(destination_path+'/MoNuSAC_processed/Images')
    except OSError:
        print ("Creation of the mask directory %s failed" % (destination_path + "\Images"))
    try:
        os.mkdir(destination_path+'/MoNuSAC_processed/Labels')
    except OSError:
        print ("Creation of the mask directory %s failed" % (destination_path + "\Labels"))
    try:
        os.mkdir(destination_path+'/MoNuSAC_processed/Overlay')
    except OSError:
        print ("Creation of the mask directory %s failed" % (destination_path + "\Overlay"))

    os.chdir(destination_path+'/MoNuSAC_processed')#Create folder named as MoNuSAC_masks
    patients = [x[0] for x in os.walk(data_path)]#Total patients in the data_path
    print('No. of Patients: ',len(patients))

    # Define Integer encoding for MoNuSAC classes (0:Background)
    nuclei_type_dict = {
            'Epithelial': 1, # ! Please ensure the matching ID is unique
            'Lymphocyte': 2,
            'Macrophage': 3,
            'Neutrophil': 4,
        }

    for patient_loc in patients:
        patient_name = patient_loc[len(data_path)+1:]#Patient name        
        """## To make patient's name directory in the destination folder
        try:
            os.mkdir(patient_name)
        except OSError:
            print ("\n Creation of the patient's directory %s failed" % patient_name)
        """    
        ## Read sub-images of each patient in the data path        
        sub_images = glob(patient_loc+'/*.svs')
        for sub_image_loc in sub_images:

            gt = 0
            sub_image_name = sub_image_loc[len(data_path)+len(patient_name)+1:-4]        
            print('File Name being processed:', sub_image_name)
            
            ## To make sub_image directory under the patient's folder
            """
            sub_image = './'+patient_name+'/'+sub_image_name #Destination path
            try:
                os.mkdir(sub_image)
            except OSError:
                print ("\n Creation of the patient's directory %s failed" % sub_image)
            """    
            image_name = sub_image_loc
            img = openslide.OpenSlide(image_name)
                                    
            # If svs image needs to save in png
            cv2.imwrite(destination_path+'/MoNuSAC_processed/Images'+sub_image_name+'.png', np.array(img.read_region((0,0),0,img.level_dimensions[0])))      
            #og_img = cv2.imread(destination_path+'/MoNuSAC_processed/Images'+sub_image_name+'.png')
            #og_img = cv2.cvtColor(og_img, cv2.COLOR_BGR2RGB)

            # Read xml file
            xml_file_name  = image_name[:-4]
            xml_file_name = xml_file_name+'.xml'
            print('XML:',xml_file_name)
            tree = ET.parse(xml_file_name)
            root = tree.getroot()

            n_ary_mask = np.transpose(np.zeros((img.read_region((0,0),0,img.level_dimensions[0]).size)))
            type_map = np.transpose(np.zeros((img.read_region((0,0),0,img.level_dimensions[0]).size)))
                          

            #Generate n-ary mask for each cell-type                         
            for k in range(len(root)):
                label = [x.attrib['Name'] for x in root[k][0]]
                label = label[0]
                
                for child in root[k]:
                    for x in child:
                        r = x.tag
                        if r == 'Attribute':
                            count = count+1
                            print(count)
                            label = x.attrib['Name']
                            class_value = nuclei_type_dict[label]
                            print(label,':',class_value)
                            
                            '''# Create directory for each label
                            sub_path = sub_image+'/'+label
                            
                            try:
                                os.mkdir(sub_path)
                            except OSError:
                                print ("Creation of the directory %s failed" % label)
                            else:
                                print ("Successfully created the directory %s " % label) 
                            '''
                            
                        if r == 'Region':
                            regions = []
                            vertices = x[1]
                            coords = np.zeros((len(vertices), 2))
                            for i, vertex in enumerate(vertices):
                                coords[i][0] = vertex.attrib['X']
                                coords[i][1] = vertex.attrib['Y']        
                            regions.append(coords)
                            poly = Polygon(regions[0]) 
                            
                            vertex_row_coords = regions[0][:,0]
                            vertex_col_coords = regions[0][:,1]
                            fill_row_coords, fill_col_coords = draw.polygon(vertex_col_coords, vertex_row_coords, n_ary_mask.shape)
                            gt = gt+1 #Keep track of giving unique valu to each instance in an image
                            n_ary_mask[fill_row_coords, fill_col_coords] = gt
                            type_map[fill_row_coords, fill_col_coords] = class_value
                            # Stack togethor the inst_map & type_map
                            
                            #nary_path = destination_path+'/MoNuSAC_processed/Labels'+sub_image_name+'_nary.tif'
                            
                            #overlay_path = destination_path+'/MoNuSAC_processed/Overlay'+sub_image_name+'.png'
                            
                            sio.savemat(destination_path+'/MoNuSAC_processed/Labels'+sub_image_name+".mat", {'inst_map':n_ary_mask,'class_map':type_map})
                            #overlay_image = visualize_instances(type_map, canvas=None)
                            #cv2.imwrite(overlay_path, overlay_image)
                            #cv2.imwrite(class_path, type_map)       


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input',help='MoNuSAC data location')
    parser.add_argument('--output',help='MoNuSAC output location')
    args = parser.parse_args()

    main(args.input,args.output)
    