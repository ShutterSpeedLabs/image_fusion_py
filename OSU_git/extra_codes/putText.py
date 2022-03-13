import cv2
import numpy as np
import os
import glob

h_down = 100

path_ir = "/media/kisna/data_1/image_fusion/image_fusion_dataset/OSU/1a/*"
path_vis = "/media/kisna/data_1/image_fusion/image_fusion_dataset/OSU/1b/*"

path_ir_list = glob.glob(path_ir, recursive=True)
path_vis_list = glob.glob(path_vis, recursive=True)
path_ir_list.sort()
path_vis_list.sort()

noOfFiles = len(path_ir_list)
img_test = cv2.imread(path_ir_list[0],-1)
# cv2.imshow('graycsale image',img_test[:,:,1])
# cv2.imshow('graycsale image',img_test)
# cv2.waitKey(0)

img_height = img_test.shape[0]
img_width = img_test.shape[1]
channels = img_test.shape[2]

imgComb = np.zeros((img_height+h_down,img_width*3,3), np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX         # font
fontScale = 0.5          # fontScale
color = (255, 255, 0)     # Red color in BGR
thickness = 1           # Line thickness of 2 px

text = "Infrared"
org = (0, 0)         # org
imgComb = cv2.putText(imgComb, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)

text = "Visible"
org = (0, 0)         # org
imgComb = cv2.putText(imgComb, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)

text = "Fused"
org = (0, 0)         # org
imgComb = cv2.putText(imgComb, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)

cv2.imshow('graycsale image',imgComb)
cv2.waitKey(0)