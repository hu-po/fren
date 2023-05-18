"""
Use v4l to get devices

sudo apt-get install v4l-utils
v4l2-ctl --list-devices
ffplay /dev/video0

v4l2-ctl -d /dev/video0 --list-formats-ext

YUYV is good for compatibility
ffplay -f video4linux2 -input_format yuyv422 -video_size 1920x1080 -framerate 30 /dev/video0

MJPG is good for individual frames
ffplay -f video4linux2 -input_format mjpeg -video_size 4096x2160 -framerate 60 /dev/video0

"""

import cv2
import time

IMG_WIDTH = 4096
IMG_HEIGHT = 2160
FPS = 60
CODEC = 'MJPG'
OUTPUT_FILE = '/home/oop/Videos/001_output.avi'
DURATION = 5

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
fourcc = cv2.VideoWriter_fourcc(*CODEC)
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

# # Show single image
# ret, image = cap.read()
# # Flip image vertically
# image = cv2.flip(image, 0)
# # Show image
# cv2.imshow('image', image)
# cv2.waitKey()

# Define the codec and create a VideoWriter object
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (IMG_WIDTH, IMG_HEIGHT))

start_time = time.time()
while int(time.time() - start_time) < DURATION:
    ret, frame = cap.read()
    if ret == True:
        # Flip the frame vertically
        frame = cv2.flip(frame, 0)
        
        # Write the flipped frame
        out.write(frame)
    else:
        break

# Release everything after the recording
cap.release()
out.release()
cv2.destroyAllWindows()