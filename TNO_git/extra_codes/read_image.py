import os
import cv2

folder_path = os.getcwd()
print(folder_path)
filename = folder_path + '/TNO_git' + "/images/IR_18rad.bmp"



img_grayscale = cv2.imread(filename,0) 

# The function cv2.imshow() is used to display an image in a window.

cv2.imshow('graycsale image',img_grayscale)

# waitKey() waits for a key press to close the window and 0 specifies indefinite loop

cv2.waitKey(0)

# cv2.destroyAllWindows() simply destroys all the windows we created.

cv2.destroyAllWindows()
