# Width = 800, Height = 600
import numpy as np
import cv2 as cv
def store(faces):

    coords = faces[:,0:4].astype(np.int32)
    score = np.array(faces[:,-1])

    return(coords,score)


### give a better score to region where faces were detected before ###
def compare(coords1,score1,coords2,score2,threshold1 = 0.7,distance = 100,bonus = 0.9):

    length1 = len(score1)
    length2 = len(score2)
    for i in range(length1):

        ### not needed if score already high
        if score1[i] > threshold1:
            # -1 is the score
            continue

        same = 0
        for j in range(length2):
            if same != 0:
                break
            else:
                if (coords1[i][0] - coords2[j][0])**2 + (coords1[i][1] - coords2[j][1])**2 < distance:
                    same = 1
                    score1[i] = bonus*score2[j] + (1-bonus)*score1[i]

def visualize_and_blur(input, coords,score, fps, threshold = 0.5):
    idx = 1
    n=len(score)
    for i in range(n):
    #if score[i] > threshold:

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

    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def clean(coords,score,threshold):
    return(coords[score>threshold],score[score>threshold])

def sort_by_score(coords,score):
    idx = score.argsort()[::-1]
    return(coords[idx],score[idx])



