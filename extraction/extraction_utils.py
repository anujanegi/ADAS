## Utility file ##
import math

# compute distance between two vectors
def distance(v1, v2):
    return ((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)

# get rectangle function
def getRectangle(im1, im2):
    x_s, x_l  = sorted([im1[0], im2[0]])
    y_s, y_l = sorted([im1[1], im2[1]])
    return [x_s, y_s, x_l-x_s+im1[2], im1[2]]

# returns True if eye is above the center axis of face
def checkEyeInFace(face, eye):
    return ((eye[1]-face[1])<(face[3]/10))

# get cropped image of best pair of eye
def getBestEyes(face, eyes):
    if(len(eyes)==0):
        # estimate the eye region in face
        eye1 = [face[2]//10, face[3]//8, 100, 100]
        eye2 = [eye1[0]+face[2]*2//3, eye1[1], 100, 100]
    elif(len(eyes)==1):
        # generate an estimate for other eye position
        eye1 = eyes[0]
        if(eye1[0]<(face[2]/2)): # ex < w/2?
            # eye found is the left eye
            eye2 = [eye1[0]+face[2]*1//3,
                         eye1[1], eye1[2], eye1[3]]
            if(eye2[0]>face[2]):
                eye2[0] = face[2]
        else:
            # eye found is the right eye
            eye2 = [eye1[0]-face[2]*1//3,
                         eye1[1], eye1[2], eye1[3]]
            if(eye2[0]<0):
                eye2[0] = 0
    elif(len(eyes)==2):
        # check for garbage values
        if(checkEyeInFace(face, eyes[0]) and checkEyeInFace(face, eyes[1])):
            # return as it is
            eye1, eye2 = eyes[0], eyes[1]
        elif checkEyeInFace(face, eyes[0]):
            return getBestEyes(face, [eyes[0]])
        elif checkEyeInFace(face, eyes[1]):
            return getBestEyes(face, [eyes[1]])
        else:
            return getBestEyes(face, [])
    else:
        # select nearest pair of eyes
        selected = [eyes[0], eyes[1]]
        min_ = distance(eyes[0], eyes[1])
        for i in range(0,len(eyes)):
            for j in range(i+1, len(eyes)):
                if(i==j):
                    continue
                else:
                    d = distance(eyes[i], eyes[j])
                    if(d<min_):
                        min_ = d
                        selected = [eyes[i], eyes[j]]
        eye1, eye2 = eyes[0], eyes[1]

    return [[eye1, eye2], getRectangle(eye1, eye2)]

# Compare two rectangles area wise
# Returns area1/area2
def compareRectangles(rec1, rec2):
    # rectangle is of the form [x,y,w,h]
    area1 = rec1[2]*rec1[3]
    area2 = rec2[2]*rec2[3]
    if area2==0:
        return math.inf
    else:
        return area1/area2
