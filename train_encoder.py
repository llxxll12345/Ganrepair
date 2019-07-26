from keras.layers import Input, Dense, Flatten, Dropout, Reshape
from keras.layers import BatchNormalization, Activation, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import TensorBoard

from keras.datasets import cifar10
import keras.backend as K
import time
from tensorboard import *

import matplotlib.pyplot as plt

import sys
import numpy as np
import os
import cv2
import argparse
from sklearn import preprocessing
from generator_conv import *

#%pylab inline

from keras.preprocessing import image

model_path = "model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"

generator = None

def show_img():
    n = 10 
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        temp = X[i] / 255
        label = Y[i] / 255
        img = temp.reshape(224, 224, 3)
        label = label.reshape(56, 56, 3)
        img[84:140, 84:140] = label
        plt.imshow(img)
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        result = generator.predict(temp.reshape(1, 224, 224, 3))
        print(img)
        #result = preprocessing.normalize([result])
        #max_val = np.max(result)
        #min_val = np.min(result)
        #result = (result - min_val) / (max_val - min_val)
        
        img[84:140, 84:140] = result.reshape(56, 56, 3)
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(img)
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def std_dev(y_true, y_pred):
    return K.sqrt(K.dot(K.transpose((y_true - y_pred)), (y_true - y_pred)))

def traingan(epochs = 20, batch_size = 32):
    # Get training images
    global X, Y
    datalen = len(X)
    trainlen = int(datalen * 0.8)
    train_X = X[:trainlen]
    train_Y = Y[:trainlen]
    test_X = X[trainlen:]
    test_Y = Y[trainlen:]
    #X_train = X_train[y_train[:,0]==1]

    #trainlen = len(train_X)
    testlen = len(test_X)

    #train = np.reshape(train, (2, trainlen, 224, 224, 3))
    #test = np.reshape(test, (2, testlen, 224, 224, 3))
    print("Training shape x:{} y:{}".format(train_X.shape, train_Y.shape))
    print("Test shape x:{} y:{}".format(test_X.shape, test_Y.shape))

    train_X = train_X / 255
    train_Y = train_Y / 255
    test_X = test_X / 255
    test_Y = test_Y / 255

    global generator

    generator, _ = myGenerator(Input(shape=(224, 224, 3)))
    
    show_img()
    if os.path.exists("generator.h5"):
        generator.load_weights("generator.h5")
    else: 
        generator.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
        generator.fit(train_X, train_Y,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(test_X, test_Y),
            callbacks=[]
        )

    generator.save('generator.h5')

    #result = generator.predict(test[:, 0])
    var = 0
    
    print("After training variance: {}".format(var))
    show_img()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=20, help="type")
    parser.add_argument("-b", "--batch_size", default=32, help="batchsize")
    args = vars(parser.parse_args())
    
    myargs = {"bad":"output", "good":"label", "epochs":args["epohs"], "batch_size":args["batchsize"]}
    files = os.listdir(myargs["bad"])
    global X, Y
    cnt = 0

    X = []
    Y = []
    temp_X = []
    temp_Y = []
    for fileName in files:
        cnt += 1
        print("Loading {}/{} ...".format(str(cnt), len(files)), end="\r")
        #os.system('clear')
        if fileName[-3:] == 'jpg' or fileName[-3:] == 'png':
            badimg = cv2.imread(os.path.join(myargs["bad"], fileName))
            goodimg = cv2.imread(os.path.join(myargs["good"], fileName))
            #print(fileName)
            temp_X.append(badimg)
            temp_Y.append(goodimg)

    #np.random.shuffle(dataset)

    idx = np.random.randint(0, len(temp_X), (len(temp_X)))
    for i in idx:
        X.append(temp_X[i])
        Y.append(temp_Y[i])

    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape)
    traingan(epochs=int(myargs["epochs"]), batch_size=int(myargs["batch_size"]))