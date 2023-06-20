import cv2 as cv
import numpy as np
import time
detector = cv.FaceDetectorYN.create(
        "yunet.onnx",
        "",
        (320, 320),
        #args.score_threshold,
        0.6
        #args.nms_threshold,
        #args.top_k
    )

img = cv.imread('data/image_test.jpg')
img_W = int(img.shape[1])
img_H = int(img.shape[0])
detector.setInputSize((img_W, img_H))

start_time = time.time()
detections = detector.detect(img)
print(time.time() - start_time)
faces = detections[1]
#print(faces)

for i in faces:
    coords = i[:-1].astype(np.int32)

    cv.rectangle(img, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 0, 255), 2)

cv.imshow('img',img)
cv.waitKey(0)


'''
cap = cv.VideoCapture('Camera_1.avi')
frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize([frameWidth, frameHeight])

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        print('No frames grabbed!')
        break

faces = detector.detect(frame)
'''