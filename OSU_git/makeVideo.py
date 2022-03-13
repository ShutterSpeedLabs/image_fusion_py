import os
import cv2

fourcc=cv2.VideoWriter_fourcc(*'MPV4)
video_out = cv2.VideoWriter('/downloads/new_video.mp4',fourcc,20,(650,490))
counter = 0
for frame in get_frames(Video_FILE):
    if frame is None:
        break
    cv2.putText(frame,text=str(counter), org=(100,100),
               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
               fontScale=3,
               color=(0,255,0),
               thickness=10)
    
    video_out.write(frame)
    counter +=1
video_out.release()