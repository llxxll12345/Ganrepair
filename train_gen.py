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
from mymobilenet import *

#%pylab inline

from keras.preprocessing import image

model_path = "model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"

generator = None
discriminator = None
gan = None

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

def get_discriminator():
    nets = MyMobileNetV2()
    conv_model = nets.MobileNetv2Conv()
    #conv_model = MobileNetV2(weights=model_path,include_top=False, input_shape=(224,224,3)) 
    #conv_model.save_weights('model/base.h5')
    
    #for i in range(144, 155):
    #    conv_model.layers.pop()
    layer = conv_model.output

    #layer = Conv2D(1280, (3, 3), padding='same', strides=(2, 2))(layer)
    #layer = BatchNormalization(axis=-1)(layer)

    #layer = Conv2D(640, (1, 1), padding='same', strides=(1, 1))(layer)
    #layer = BatchNormalization(axis=-1)(layer)

    layer = Activation('relu')(layer)
    layer = GlobalAveragePooling2D()(layer)
    #x = Dense(512,activation='relu')(x) 
    #x = Dropout(0.25)(x)
    layer = Dense(256,activation='relu')(layer) 
    layer = Dropout(0.5)(layer)
    
    outlayer=Dense(1, activation='softmax')(layer)
     
    model=Model(inputs=conv_model.input,outputs=outlayer)
    model.summary()

    return model, outlayer

def get_model():
    discriminator, disc_out = get_discriminator()
    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
    #discriminator.trainable = False

    ganInput = Input(shape=(224, 224, 3))
    generator, ganOut = myGenerator(ganInput)

    ganOutput = discriminator(ganOut)
    gan = Model(ganInput, ganOutput)
    gan.summary()

    gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
    return discriminator, generator, gan


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

    global generator, discriminator, gan

    discriminator, generator, gan = get_model()
    
    if os.path.exists("model/gan.h5"):
        generator.load_weights("model/gan.h5")
        show_img()
        return

    num_batches = int(trainlen/batch_size)
    last_d_loss = 0.0
    last_g_loss = 0.0
    for epoch in range(epochs):
        cum_d_loss = 0.0
        cum_g_loss = 0.0
        timespent = 0.0
        for batch_idx in range(num_batches):
            if batch_idx != 0:
                print("Now: {}/{}, Time spent: {}s Est time:{}s".format(batch_idx, num_batches, timespent, \
                    round(timespent * (num_batches - batch_idx), 2)))
            inputs =  train_X[batch_idx * batch_size : (batch_idx + 1) * batch_size] # bad image
            outputs = train_Y[batch_idx * batch_size : (batch_idx + 1) * batch_size] # good image

            start = time.time()
            generated_images = generator.predict(inputs)

            # Train on soft labels (add noise to labels as well)
            noise_prop = 0.05 # Randomly flip 5% of labels
            
            # Prepare labels for real data（good image)
            true_labels = np.zeros((batch_size, 1)) + np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
            flipped_idx = np.random.choice(np.arange(len(true_labels)), size=int(noise_prop * len(true_labels)))
            true_labels[flipped_idx] = 1 - true_labels[flipped_idx]
            
            # Train discriminator on real data (good image)
            d_loss_true = discriminator.train_on_batch(outputs, true_labels)

            # Prepare labels for generated data 
            gene_labels = np.ones((batch_size, 1)) - np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
            flipped_idx = np.random.choice(np.arange(len(gene_labels)), size=int(noise_prop*len(gene_labels)))
            gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]
            
            # Train discriminator on generated data
            d_loss_gene = discriminator.train_on_batch(generated_images, gene_labels)

            d_loss = 0.5 * np.add(d_loss_true, d_loss_gene)
            cum_d_loss += d_loss

            # Train generator
            g_loss = gan.train_on_batch(inputs, np.zeros((batch_size, 1)))
            cum_g_loss += g_loss

            timespent = round((time.time() - start), 2)

        cur_g_loss = cum_g_loss/num_batches
        cur_d_loss = cum_d_loss/num_batches
        print('Epoch: {}, Generator Loss: {}, Discriminator Loss: {}'.format(epoch + 1, cur_g_loss, cur_d_loss))
        #show_imgs("epoch" + str(epoch))
       
        show_img()
        # early stopping
        #if abs(cur_d_loss - last_d_loss) < 0.01 or abs(cur_g_loss - last_g_loss) < 0.01:
        #    break
        last_d_loss = cur_d_loss
        last_g_loss = cur_g_loss

        variance = 0
        for a, b in zip(test_X, test_Y):
            result = generator.predict(a)
            variance += (result - b) * (result - b)
        
        print('Average variance: {}'.format(round(variance / testlen, 3)))

    generator.save("model/generator.h5")
    gan.save("model/gan.h5")
    discriminator.save("model/discriminator.h5")
    

if __name__ == "__main__":
    args = {"bad":"output", "good":"label", "epochs":20, "batch_size":32}
    files = os.listdir(args["bad"])
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
            badimg = cv2.imread(os.path.join(args["bad"], fileName))
            goodimg = cv2.imread(os.path.join(args["good"], fileName))
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
    traingan(epochs=int(args["epochs"]), batch_size=int(args["batch_size"]))