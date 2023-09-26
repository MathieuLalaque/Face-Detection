import argparse
from functions_tr import *
from pyautogui import size

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()

parser.add_argument('--video', '-v', type=str, help='Path to the input video or rstp url.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='yunet.onnx',
                    help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--score_threshold','-st', type=float, default=0.3,
                    help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold','-n',type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--use_web_cam','-uwc', type=str, default='false', help='Use built in webcam')
args = parser.parse_args()


def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't','True']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f','False']:
        return False
    else:
        raise NotImplementedError

# Load cam into seperate process
print("Cam loading")

if str2bool(args.use_web_cam):
    cap = cv.VideoCapture(0)
else:
    cap = cv.VideoCapture(args.video, cv.CAP_FFMPEG)



print("Cam loaded")

detector = cv.FaceDetectorYN.create(
    args.face_detection_model,
    "",
    (320, 320),
    args.score_threshold,
    args.nms_threshold,
    args.top_k,
)

tm = cv.TickMeter()

true_frameWidth = cap.get(cv.CAP_PROP_FRAME_WIDTH)
true_frameHeight = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

screenWidth, screenHeight = size()

width_ratio = screenWidth/true_frameWidth
height_ratio = screenHeight/true_frameHeight


if width_ratio > height_ratio:
    cv.namedWindow('Live', cv.WINDOW_NORMAL)
    cv.resizeWindow('Live', int(true_frameWidth*height_ratio), screenHeight)
else:
    cv.namedWindow('Live', cv.WINDOW_NORMAL)
    cv.resizeWindow('Live', screenWidth, int(true_frameHeight*width_ratio))




frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) * args.scale)
frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) * args.scale)
detector.setInputSize([frameWidth, frameHeight])



### initiate useful variables
latence = create_latence((frameHeight, frameWidth))
max = 0
count = 0
###
try:
    while cv.waitKey(1) < 0:
        # Grap frame from the buffer

        #cap.grab()

        hasFrame, or_frame = cap.read()
        tm.start()
        # Process frame here

        if not hasFrame:
            print('No frames grabbed!')
            continue
        frame = cv.resize(or_frame, (frameWidth, frameHeight))
        # Inference
        faces = detector.detect(frame)  # faces is a tuple

        # Draw results on the input image
        #
        # modification
        #
        # visualize(frame, faces, tm.getFPS())
        #
        #
        #
        if faces[1] is not None:
            coords, score = store(faces[1])
            coords, score = sort_by_score(coords, score)

            max = visualize_and_blur(or_frame, coords, score, tm.getFPS(), latence, max, scale=args.scale,threshold2=args.score_threshold,n_latence=30)

        latence -= 1
        count += 1

        if count % 900 == 0:
            tm.reset()
            max = 0

        # Visualize results

        cv.imshow('Live', or_frame)

        tm.stop()

# If close requested
except KeyboardInterrupt:

    cap.release()
    cv.destroyAllWindows()
    print("Camera connection closed")