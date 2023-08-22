import numpy as np
import cv2 as cv


def create_latence(size = (600,800)):

    """ Create numpy array the size of the images to store the detection of high confidence faces

    :param size: size of the image, /!\ (y,x) format /!\
    :type size: tuple(int,int)
    """
    return(np.zeros(size,np.int32))

def ping(x,y,latence,size_latence,n_latence =100):
    """ Update the latence array with a high confidence face

    This method takes the coordinates of a high condience face and update the latence array
    by assigning the value n_latence to a square of middle (x,y) and side size_latence/2

    :param x,y: coordinates of the face
    :type x,y: int
    :param latence: latence array
    :type latence: numpy array
    :param size_latence : size of the area in which the score threshold is lowered
    :type size_latence : int
    :param n_latence : number of frames the treshold will be lowered
    :type n_latence: int
    """
    size = round(size_latence)
    latence[y-size:y+size,x-size:x+size] = n_latence

def check_latence(x,y,latence):

    "Check if the threshold is lowered for the (x,y) coordinates"
    return(latence[y,x] > 0)

def store(faces):

    """Put the output of the Yunet model in two arrays, coords for the coordinates, score for the confidence score

    :param faces: output of the model
    :type faces: numpy array
    """

    coords = faces[:,0:4].astype(np.int32)
    score = np.array(faces[:,-1])

    return(coords,score)

def sort_by_score(coords,score):

    """Sort faces according to their confidence score in decreasing order

    :param coords,score: output of the store function

    """
    idx = score.argsort()[::-1]
    return(coords[idx],score[idx])
def visualize_and_blur(input, coords,score, fps,latence,max,size_latence=10,threshold = 0.6,threshold2 = 0.15,scale = 1.0):

    """Takes the current frame as input and blur the relevant areas

    This method take the current frame and blur the relevant areas and update useful variables for the next frames

    :param input: current frame
    :type input: numpy array
    :param coords,score : output from store and sort_by_score functions
    :type coords,score : numpy array
    :param fps: frames processed by the algorithm per seconds
    :type fps: float
    :param latence: latence array
    :type latence: numpy array
    :param max: maximum number of faces detected with a high confidence
    :type max: int
    :param size_latence : size of the area in which the score threshold is lowered
    :type size_latence : int
    :param threshold : threshold from which  it is considered a face is detected with high confidence (if higher)
    :type threshold: float
    :param threshold2: threshold from which output of the models are considered to be faces IF faces were detected before (if higher)
    :type threshold2: float

    """
    i = 0
    ratio = 1/scale

    n=len(score)
    # Faces with high confidence
    while i < n and score[i] > threshold:

        print(
            'Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
                i+1, coords[i][0], coords[i][1], coords[i][2], coords[i][3], score[i]))

        x = coords[i][0]
        y = coords[i][1]
        w = coords[i][2]
        h = coords[i][3]

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        #blur the area

        ROI = input[round(ratio*y):round(ratio*(y + h)), round(ratio*x):round(ratio*(x + w))]

        #blurring not strong enough if the face is big
        if w*ratio > 70 or h*ratio > 70:
            blur = cv.blur(ROI, (25, 25))
        else:
            blur = cv.blur(ROI,(10,10))



        # Insert ROI back into image
        input[round(ratio*y):round(ratio*(y + h)), round(ratio*x):round(ratio*(x + w))] = blur

        ping(x,y,latence,size_latence*scale)

        i+=1
    #update max number of faces
    if i > max:
        new_max = i
    else:
        new_max = max

    idx = i

    # Faces with low confidence
    while idx < new_max + 2 and i < n and score[i] > threshold2:

        x = coords[i][0]
        y = coords[i][1]

        #model can predict negative values when face partially outside screen
        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if check_latence(x,y,latence):

            idx+=1

            w = coords[i][2]
            h = coords[i][3]
            # print(x,y,w,h)

            ROI = input[round(ratio*y):round(ratio*(y + h)), round(ratio*x):round(ratio*(x + w))]
            if w*ratio > 70 or h*ratio > 70:
                blur = cv.blur(ROI, (25, 25))
            else:
                blur = cv.blur(ROI, (10, 10))

            # Insert ROI back into image
            input[round(ratio * y):round(ratio * (y + h)), round(ratio * x):round(ratio * (x + w))] = blur


            print(
                'Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
                    idx, coords[i][0], coords[i][1], coords[i][2], coords[i][3], score[i]))
        i+=1

    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return(new_max)

def predict(coords,score,latence,max,size_latence=10,threshold = 0.6,threshold2 = 0.15,scale = 1.0):

    """Takes the current frame as input and blur the relevant areas

    This method take the current frame and blur the relevant areas and update useful variables for the next frames

    :param coords,score : output from store and sort_by_score functions
    :type coords,score : numpy array
    :param fps: frames processed by the algorithm per seconds
    :type fps: float
    :param latence: latence array
    :type latence: numpy array
    :param max: maximum number of faces detected with a high confidence
    :type max: int
    :param size_latence : size of the area in which the score threshold is lowered
    :type size_latence : int
    :param threshold : threshold from which  it is considered a face is detected with high confidence (if higher)
    :type threshold: float
    :param threshold2: threshold from which output of the models are considered to be faces IF faces were detected before (if higher)
    :type threshold2: float

    """
    i = 0


    n=len(score)
    # Faces with high confidence
    while i < n and score[i] > threshold:

        #print(
        #    'Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
        #        i+1, coords[i][0], coords[i][1], coords[i][2], coords[i][3], score[i]))

        x = coords[i][0]
        y = coords[i][1]

        ping(x,y,latence,size_latence*scale)

        i+=1
    #update max number of faces
    if i > max:
        new_max = i
    else:
        new_max = max

    idx = i
    liste =[ a for a in range(i+1)]

    # Faces with low confidence
    while idx < new_max + 5 and i < n and score[i] > threshold2:

        x = coords[i][0]
        y = coords[i][1]

        if check_latence(x,y,latence):

            idx+=1
            liste.append(i)



            #print(
            #    'Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
            #        idx, coords[i][0], coords[i][1], coords[i][2], coords[i][3], score[i]))
        i+=1

    liste = np.array(liste)

    prediction = np.zeros((len(liste),5))
    prediction[:,:4] = coords[liste][:,np.array([0,1,2,3])]
    # we set the confidence level to the threshold in case it is lower than the threshold
    prediction[:,4] = np.where(score[liste] < threshold,threshold,score[liste])

    return(new_max,prediction)




