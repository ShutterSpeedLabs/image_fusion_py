import os
import cv2

folder_path = os.getcwd()
print(folder_path)
filename = folder_path + '/OSU_git' + '/images/img_00000.bmp'

img = cv2.imread(filename,cv2.IMREAD_UNCHANGED) 

imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

cv2.imshow('graycsale image',imgYCC)

# waitKey() waits for a key press to close the window and 0 specifies indefinite loop

cv2.waitKey(0)

imgYCC = cv2.cvtColor(imgYCC, cv2.COLOR_YCrCb2BGR)

cv2.imshow('graycsale image',imgYCC)

# waitKey() waits for a key press to close the window and 0 specifies indefinite loop

cv2.waitKey(0)
# cv2.destroyAllWindows() simply destroys all the windows we created.

cv2.destroyAllWindows()