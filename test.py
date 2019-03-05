
import cv2
import os

vpath='/home/urhgyt/videos/ricks2e5.mkv'

cap  = cv2.VideoCapture(vpath)
print cap.isOpened()

frame_count = 1
success = True
#img = cv2.imread('/home/urhgyt/videos/video_00005.jpg', -1)
while (success):
    success, frame = cap.read()
    print 'Read a new frame: ', success

    params = []
    # params.append(cv.CV_IMWRITE_PXM_BINARY)
    params.append(1)
    print frame.shape
#    rows,cols,chennels = img.shape
    frame=cv2.resize(frame,(720, 480))
    frame=frame[112:368,232:488]
    cv2.imwrite("/home/urhgyt/videos/"+"video" + "_%05d.jpg" % frame_count, frame, params)
    print frame_count
    frame_count = frame_count + 1

cap.release()