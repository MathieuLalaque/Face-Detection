def convert_to_yolo_confidence(x,y,w,h,frame_width,frame_heigth,confidence):
    x_center = (x +w/2)/frame_width
    y_center = (y + h/2)/frame_heigth
    width = w/frame_width
    height = h/frame_heigth
    label = ("0",str(confidence),str(x_center),str(y_center),str(width),str(height),'\n')
    return(" ".join(label))


#############################"



import argparse
import numpy as np
import cv2 as cv
from functions_tr import *
import os

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError


parser = argparse.ArgumentParser()
parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
parser.add_argument('--image2', '-i2', type=str,
                    help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='face_detection_yunet_2021dec.onnx',
                    help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='face_recognition_sface_2021dec.onnx',
                    help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold','-st', type=float, default=0.9,
                    help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold','-n',type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False,
                    help='Set true to save results. This flag is invalid when using camera.')
args = parser.parse_args()


def create_prediction(path,output_folder, scale = 1, face_detection_model = 'yunet.onnx', score_threshold = 0.149, nms_threshold=0.3, top_k=5000):

    detector = cv.FaceDetectorYN.create(
        face_detection_model,
        "",
        (320, 320),
        score_threshold,
        nms_threshold,
        top_k
    )

    cap = cv.VideoCapture(path)
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) * scale)
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) * scale)
    detector.setInputSize([frameWidth, frameHeight])

    #variables to create database
    list = np.linspace(30, cap.get(cv.CAP_PROP_FRAME_COUNT), 10, dtype=int)
    img_idx = 0

    ### initiate useful variables
    latence = create_latence((frameHeight,frameWidth))
    max = 0
    count = 0
    total_frames = str(cap.get(cv.CAP_PROP_FRAME_COUNT))
    ###

    while cv.waitKey(1) < 0:
        #print('Frame numéro '+str(cap.get(cv.CAP_PROP_POS_FRAMES)))
        hasFrame, or_frame = cap.read()
        frame_index = cap.get(cv.CAP_PROP_POS_FRAMES)


        if not hasFrame:
            print('No frames grabbed!')
            continue
        frame = cv.resize(or_frame, (frameWidth, frameHeight))
        # Inference
        faces = detector.detect(frame)  # faces is a tuple

        if faces[1] is not None:
            coords, score = store(faces[1])
            coords,score = sort_by_score(coords,score)

            max,prediction = predict(coords,score,latence,max,scale = scale)

        latence -= 1
        count +=1

        if (count%900) == 0:
            print(str(frame_index) + '/' + str(total_frames))
            max = 0

        if frame_index== list[img_idx]:

            faces_liste = []
            if faces[1] is not None:
                for face in prediction:
                    faces_liste.append(convert_to_yolo_confidence(face[0], face[1], face[2], face[3], frameWidth, frameHeight,face[4]))
            with open(os.path.join(output_folder, 'img_' + str(img_idx) + '.txt'), "w") as file:
                file.writelines(faces_liste)

            img_idx +=1


#create_prediction("data/estw/Camera_21.avi",'output/predictions/estw/Camera_21')