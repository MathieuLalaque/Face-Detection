import cv2 as cv
import numpy as np
import os
def convert_to_yolo(x,y,w,h,frame_width,frame_heigth):
    x_center = (x +w/2)/frame_width
    y_center = (y + h/2)/frame_heigth
    width = w/frame_width
    height = h/frame_heigth
    label = ("0",str(x_center),str(y_center),str(width),str(height),'\n')
    return(" ".join(label))


def create_gt(file_path,output_folder):
    cap = cv.VideoCapture(file_path)

    print('frame rate is '+str(cap.get(cv.CAP_PROP_FPS)))
    print('total number of frames is ' +str(cap.get(cv.CAP_PROP_FRAME_COUNT)))
    length = cap.get(cv.CAP_PROP_FRAME_COUNT)/cap.get(cv.CAP_PROP_FPS)
    print('durée de la vidéo: '+str(int(length//60))+'mn'+str(int(length%60)))

    list = np.linspace(30,cap.get(cv.CAP_PROP_FRAME_COUNT),10 ,dtype=int)

    detector = cv.FaceDetectorYN.create(
            'yunet.onnx',
            "",
            (320, 320))

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])





    count = 0
    while cv.waitKey(1) < 0 and count<10:
        cap.set(cv.CAP_PROP_POS_FRAMES,list[count])
        hasFrame, frame = cap.read()
        if hasFrame:
            faces = detector.detect(frame)
            #print(faces)
            cv.imshow('Live', frame)
            cv.imwrite(os.path.join(output_folder,'img_'+str(count)+'.jpg'),frame)

            faces_liste = []

            try:

                for face in faces[1]:
                    faces_liste.append(convert_to_yolo(face[0],face[1],face[2],face[3],frameWidth,frameHeight))

                with open(os.path.join(output_folder,'img_'+str(count)+'.txt'),"w") as file:
                    file.writelines(faces_liste)
            except TypeError:
                print('no face detected')
            except Exception as e:
                print("something else went wrong")
                print(e)
        count+=1
    cv.destroyAllWindows()

#create_gt('data/Michiel_data/Camera_4.avi')