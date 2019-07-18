import copy
import cv2
import numpy as np

def data_augment(img = None, path = None): #augment=True):
    #assert 'filepath' in img_data
    #assert 'bboxes' in img_data
    #assert 'width' in img_data
    #assert 'height' in img_data

    try:
        if path != None:
            img = cv2.imread(path)
    except:
        print("Error parameters")
        return None

    rows, cols = img.shape[:2]

    # Adding randomness to the horizontal flip 
    if np.random.randint(0, 2) == 0:
        # left and right
        img = cv2.flip(img, 1)

    if np.random.randint(0, 2) == 0:
        # up and down
        img = cv2.flip(img, 0)


    if np.random.randint(0, 2) == 0:
        # choose one from the four angles
        angle = np.random.choice([0, 90, 180, 270], 1)[0]
        if angle == 90: 
             # -90 transpose, then flip up down
            img = np.transpose(img, (1, 0, 2))
            img = cv2.flip(img, 0)
        elif angle == 180:
            # upside down and left right
            img = cv2.flip(img, -1)
        elif angle == 270:
            # -90 transpose, then flip left right
            img = np.transpose(img, (1, 0, 2))
            img = cv2.flip(img, 1)
        elif angle == 0:
            pass
            
    return img
