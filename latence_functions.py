import numpy as np
from spar import spar
import cv2 as cv

#id_list = range(50)
#id_threshold = np.zeros((50,100))

def init_partition(n,size =(600,800)):

    part_width = np.linspace(0, size[1], n + 1).astype(np.int32)
    part_height = np.linspace(0, size[0], n + 1).astype(np.int32)

    partition = np.zeros((size[0], size[1]))
    idx = 0
    for i in range(n):
        for j in range(n):
            partition[part_height[i]:part_height[i + 1], part_width[j]:part_width[j + 1]] = idx
            idx += 1

    partition = spar(partition)
    return(partition)

def create_latence(n):
    return(np.zeros(n,np.int32))

def ping(x,y,partition,latence,n_latence =100):
    latence[partition[y,x]] = n_latence

def check_latence(x,y,partition,latence):
    return(latence[partition[y,x]] > 0)


def visualize_and_blur(input, coords,score, fps,partition,latence,threshold = 0.8,threshold2 = 0.15):
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

            ping(x,y,partition,latence)

        else:
            if score[i] > threshold2:
                x = coords[i][0]
                y = coords[i][1]

                if check_latence(x,y,partition,latence):
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

def draw_lines(input,n,size =(600,800)):
    part_width = np.linspace(0, size[1], n + 1).astype(np.int32)
    part_height = np.linspace(0, size[0], n + 1).astype(np.int32)
    img = input

    for i in part_height:
        img = cv.line(img, (0, i), (799, i), (0, 0, 255), thickness=1)
    for i in part_width:
        img = cv.line(img, (i, 0), (i, 599), (0, 0, 255), thickness= 1)

    return(img)

'''
def check_before()
'''