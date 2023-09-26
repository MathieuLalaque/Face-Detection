import argparse
import numpy as np
import cv2 as cv
from functions_tr import *

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
parser.add_argument('--face_detection_model', '-fd', type=str, default='yunet.onnx',
                    help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='face_recognition_sface_2021dec.onnx',
                    help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold','-st', type=float, default=0.2,
                    help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold','-n',type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False,
                    help='Set true to save results. This flag is invalid when using camera.')
args = parser.parse_args()

if __name__ == '__main__':

    detector = cv.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )

    tm = cv.TickMeter()

    if args.video is not None:
        deviceId = args.video
    else:
        deviceId = 0
    cap = cv.VideoCapture(deviceId)
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) * args.scale)
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) * args.scale)
    detector.setInputSize([frameWidth, frameHeight])

    ### initiate useful variables
    latence = create_latence((frameHeight,frameWidth))
    max = 0
    count = 0
    ###

    while cv.waitKey(1) < 0:
        hasFrame, or_frame = cap.read()
        tm.start()
        if not hasFrame:
            print('No frames grabbed!')
            break
        frame = cv.resize(or_frame, (frameWidth, frameHeight))
        # Inference
        faces = detector.detect(frame)  # faces is a tuple

        # Draw results on the input image
        #
        #modification
        #
        #visualize(frame, faces, tm.getFPS())
        #
        #
        #
        if faces[1] is not None:
            coords, score = store(faces[1])
            coords,score = sort_by_score(coords,score)

            max = visualize_and_blur(or_frame,coords,score,tm.getFPS(),latence,max,scale = args.scale,threshold2=args.score_threshold)

        latence -= 1
        count +=1

        if count%900 == 0:
            tm.reset()
            max = 0

        # Visualize results

        cv.imshow('Live', or_frame)
        tm.stop()

    cv.destroyAllWindows()
    cv.release()