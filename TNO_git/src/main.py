import cv2
import numpy as np
import os
import glob

path_1a = "/media/kisna/data_1/image_fusion/image_fusion_dataset/OSU/1a/*"
path_1b = "/media/kisna/data_1/image_fusion/image_fusion_dataset/OSU/1b/*"

path_1a_list = glob.glob(path_1a, recursive=True)
path_1b_list = glob.glob(path_1b, recursive=True)

noOfFiles = len(path_1a_list)

for fileNo in range(noOfFiles):
    print(path_1a_list[fileNo])
    print(path_1b_list[fileNo])        
