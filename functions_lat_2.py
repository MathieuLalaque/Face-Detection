import numpy as np
import cv2 as cv


def create_latence(size = (600,800)):
    return(np.zeros(size,np.int32))

def ping(x,y,latence,size_latence,n_latence =100):

#for now no check if on edge
    latence[y-size_latence:y+size_latence,x-size_latence:x+size_latence] = n_latence

def check_latence(x,y,latence):
    return(latence[y,x] > 0)


def visualize_and_blur(input, coords,score, fps,latence,size_latence=10,threshold = 0.8,threshold2 = 0.15):
    idx = 1
    n=len(score)
    for i in range(n):
        if score[i] > threshold:

            print(
                'Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
                    idx, coords[i][0], coords[i][1], coords[i][2], coords[i][3], score[i]))

            x = coords[i][0]
            y = coords[i][1]
            w = coords[i][2]
            h = coords[i][3]
            #print(x,y,w,h)

            ROI = input[y:y + h, x:x + w]
            blur = cv.blur(ROI,(10,10))

            # Insert ROI back into image
            input[y:y + h, x:x + w] = blur
            idx+=1

            ping(x,y,latence,size_latence)

        else:
            if score[i] > threshold2:
                x = coords[i][0]
                y = coords[i][1]

                if check_latence(x,y,latence):
                    w = coords[i][2]
                    h = coords[i][3]
                    # print(x,y,w,h)

                    ROI = input[y:y + h, x:x + w]
                    blur = cv.blur(ROI, (10, 10))

                    # Insert ROI back into image
                    input[y:y + h, x:x + w] = blur
                    idx += 1

                    print(
                        'Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
                            idx, coords[i][0], coords[i][1], coords[i][2], coords[i][3], score[i]))

    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)






def store(faces):

    coords = faces[:,0:4].astype(np.int32)
    score = np.array(faces[:,-1])

    return(coords,score)

def sort_by_score(coords,score):
    idx = score.argsort()[::-1]
    return(coords[idx],score[idx])