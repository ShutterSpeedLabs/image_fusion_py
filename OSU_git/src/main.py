import cv2
import numpy as np
import os
import glob

h_down = 40


   
def applySobel(img_gray):
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3) # Combined X and Y Sobel Edge Detection
    return sobelxy

def applykernel(image):
    kernel2 = np.ones((5, 5), np.float32) / 25
    img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)
    return img

def framesCombineText(img_irIn,img_visIn,img_fusedIn):
    img_height = img_irIn.shape[0]
    img_width = img_irIn.shape[1]
    channels = img_irIn.shape[2]
    imgComb = np.zeros((img_height+h_down,img_width*3,3), np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX         # font
    fontScale = 0.5          # fontScale
    color = (0, 255, 0)     # Red color in BGR
    thickness = 1           # Line thickness of 2 px

    text = "Infrared"
    org = (120, 20)         # org
    imgComb = cv2.putText(imgComb, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)

    text = "Visible"
    org = (440, 20)         # org
    imgComb = cv2.putText(imgComb, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)

    text = "Fused"
    org = (760, 20)         # org
    imgComb = cv2.putText(imgComb, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)

    imgComb[h_down-1:-1,0:img_width,:] = img_ir
    imgComb[h_down-1:-1,img_width:2*img_width,:] = img_vis
    imgComb[h_down-1:-1,2*img_width:3*img_width,:] = imgrgb 

    return imgComb


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

fused_image = np.zeros((img_height,img_width,3), np.uint8)
fused_video = np.zeros((img_height+h_down,img_width*3,3,noOfFiles), np.uint8)


print("image height is: ", img_height)
print("Image width is:", img_width)
print("Number of channels are: ", channels)
i=0
for fileNo in range(noOfFiles):
    print(path_ir_list[fileNo])
    print(path_vis_list[fileNo]) 
    img_ir = cv2.imread(path_ir_list[fileNo],cv2.IMREAD_UNCHANGED) 
    img_vis = cv2.imread(path_vis_list[fileNo],cv2.IMREAD_UNCHANGED) 
    img_vis_y = cv2.cvtColor(img_vis, cv2.COLOR_RGB2YUV)
    img_ir_hi = applySobel(img_ir[:,:,0])
    img_ir_lo = applykernel(img_ir[:,:,0])
    img_vis_hi = applySobel(img_vis_y[:,:,0])
    img_vis_lo = applykernel(img_vis_y[:,:,0])
    for i in range(img_height):
        for j in range(img_width):
            alpha1 = max((img_ir_hi[i,j]+img_ir_lo[i,j]),0)
            alpha2 = max((img_ir_hi[i,j]+img_ir_lo[i,j]+img_vis_hi[i,j]+img_vis_lo[i,j]+1),1)
            alpha = alpha1/alpha2
            beta = 1-alpha
            fused_image[i,j,0] = alpha*img_ir[i,j,0] + beta*img_vis_y[i,j,0]
            fused_image[i,j,1] = img_vis_y[i,j,1]
            fused_image[i,j,2] = img_vis_y[i,j,2]
    imgrgb = cv2.cvtColor(fused_image, cv2.COLOR_YUV2RGB)    
    fused_video[:,:,:,i] = framesCombineText(img_ir,img_vis,imgrgb)
    cv2.imshow('graycsale image',fused_video[:,:,:,i])
    cv2.waitKey(2)
    i = i+1
