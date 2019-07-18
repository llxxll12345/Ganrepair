import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import copy
import argparse
from data_augment import *

c_x = 0
c_y = 0

def generate_noise_picture(sz):
    img = np.random.normal(-0.1, 0.1, size=(sz, sz, 3))
    return img

def genCircle(img, woffset, hoffset, width, height):
    px = np.random.randint(-width / 4, width / 4, size=1)  + woffset / 2
    py = np.random.randint(-height / 4, height / 4, size=1) + hoffset / 2
    r = np.random.randint(-width / 20, width / 20, size=1) + min(width, height) / 4
    cv2.circle(img, (int(px), int(py)), int(r + 5), (5, 5, 5), -1, 8)

def generate_reflection(bg_dir, fg_dir):
    pass

def draw(imgName, img, cnt):
    height, width = img.shape[:2]
    print("Image:{0}: {1},{2}".format(imgName, width, height))
    #cv2.namedWindow(imgName) 
    #cv2.imshow(imgName,img)
    #cv2.waitKey(0)
    output = copy.deepcopy(img)
    output= cv2.resize(output, (224, 224), cv2.INTER_CUBIC)
    cv2.rectangle(output, (84, 84), (139, 139), (5, 5, 5), -1, 8)

    img = cv2.resize(img, (224, 224), cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join("output", "{}".format(cnt) + ".jpg"), output)

    label = img[84:140, 84:140]
    cv2.imwrite(os.path.join("label", "{}".format(cnt) + ".jpg"), label)

    #cv2.imshow(imgName, output)
    #cv2.waitKey(0)

    #output = data_augment(img = output, path = None)

    '''
    output = output.astype(np.float64)
    output /= 255.0
    noise = generate_noise_picture(224)
    output += noise
    output *= 255.0
    cv2.imwrite(os.path.join("output", "{}-aug".format(cnt) + ".jpg"), output)
    cv2.imwrite(os.path.join("label", "{}-aug".format(cnt) + ".jpg"), label)
    '''

    output[84:140, 84:140] = label

    #cv2.imshow(imgName, output)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def processImg(dir, type, fgd=None):
    cnt = 1
    if type == 1:
        dirs = os.listdir(dir)
        print(dirs)
        for diry in dirs:
            if diry == ".DS_Store":
                continue
            dirname = os.path.join(dir, diry)
            files = os.listdir(dirname)
            file_cnt = len(files)
            for fileName in files:
                print(fileName)
                if fileName[-3:] != 'jpg' and fileName[-3:] != 'png':
                    continue
                img = cv2.imread(os.path.join(dirname, fileName))
                cnt += 1
                print("Image {0}/{1}: {2}".format(cnt, file_cnt, fileName))
                draw(fileName.split('/')[-1], img, cnt)
    if type == 2:
        pass
    print("Fin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", required=True, help="directory")
    parser.add_argument("-f", "--fgd", default=None, help="foreground")
    parser.add_argument("-t", "--type", default=1, help="type")
    args = vars(parser.parse_args())
    processImg(args["dir"], args["type"], args["fgd"])


if __name__ == '__main__':
    main()
    
        
