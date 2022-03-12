import os
import cv2

folder_path = os.getcwd()
print(folder_path)
filename = folder_path + '/TNO_git' + "/images/IR_18rad.bmp"



img_grayscale = cv2.imread(filename,0) 


imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
