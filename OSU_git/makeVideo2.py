w = img_width
h = img_height
fps = 20
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter('output_video.avi', fourcc, fps, (w, h))

for i in range(noOfFiles):
    image_n = fused_video[:,:,:,i]
    writer.write(image_n)    
    cv2.imshow('graycsale image',image_n) 
    cv2.waitKey(0)

writer.release()