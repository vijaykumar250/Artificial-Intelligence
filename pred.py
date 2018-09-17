from keras.models import model_from_yaml
import data as dg
import numpy as np
from keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
from random import randrange
from PIL import Image
import pandas as pd
import cv2

def model_loader(model_name="", file_path = "E:\Machine Learning\color_sep\Models"):
    if model_name != "":
        # load YAML and create model
        yaml_file = open(file_path + os.sep + model_name + ".yaml", 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        loaded_model.load_weights(file_path + os.sep + model_name + ".h5")
        print("Loaded model from disk")

        return loaded_model
        # # evaluate loaded model on test data
        # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # score = loaded_model.evaluate(X, Y, verbose=0)
        # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

def run_random_imges():
    train_images, train_labels, test_images, test_labels = dg.load_data(split_ratio=0.8,total_row=100)

    # test_images.shape will return no. of images, width, height, channel info
    # test_labels.shape will return the total no. of test data
    print('Testing data shape : ', test_images.shape, test_labels.shape)
    print(test_labels)

    # Find the unique numbers from the train labels
    classes = np.unique(test_labels)
    nClasses = len(classes)
    print('Total number of outputs : ', nClasses)   #0, 1 for red, green, 2 classes
    print('Output classes : ', classes)             #classes are [0,1]

    # ==============================================================================
    # Preprocess the Data
    # Find the shape of input images and create the variable input_shape
    nRows, nCols, nDims = test_images.shape[1:]
    test_data = test_images.reshape(test_images.shape[0], nRows, nCols, nDims)

    # Change to float datatype
    test_data = test_data.astype('float32')

    # Scale the data to lie between 0 to 1
    test_data /= 255

    # Change the labels from integer to categorical data
    test_labels_one_hot = to_categorical(test_labels)

    # Display the change for category label using one-hot encoding
    print('Original label 0 : ', test_labels)
    print('After conversion to categorical ( one-hot ) : ', test_labels_one_hot)

    model_name = "Model1_20180917_100332"
    model = model_loader(model_name=model_name)
    # # evaluate loaded model on test data
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # score = model.evaluate(test_images, test_labels_one_hot, verbose=1)
    # print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    test_image = cv2.imread('E:\Machine Learning\color.png')
    print('img: ',test_image)
    # test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image = cv2.resize(test_image,(32,32))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255
    print('new image:',test_image.shape)

    test_image1 = cv2.imread('E:\Machine Learning\color1.png')
    print('img: ',test_image)
    # test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image1 = cv2.resize(test_image1, (32, 32))
    test_image1 = np.array(test_image1)
    test_image1 = test_image1.astype('float32')
    test_image1 /= 255
    print('new image 1:', test_image1.shape)

    if len(np.array(test_image).shape) == 2:
        channel = 1
    #color image
    else:
        channel = np.array(test_image).shape[2]

    test_image = np.expand_dims(test_image,axis=0)
    print("image shape :",test_image.shape)

    if len(np.array(test_image1).shape) == 2:
        channel = 1
    #color image
    else:
        channel = np.array(test_image1).shape[2]

    test_image1 = np.expand_dims(test_image1,axis=0)
    print("image shape 1: ",test_image1.shape)

    print('model predict :',model.predict(test_image))
    prediction = model.predict_classes(test_image)
    print("prediction:", prediction)

    print('model predict 1 :', model.predict(test_image1))
    prediction = model.predict_classes(test_image1)
    print("prediction 1:", prediction)

    print("Actual:    ", test_labels.reshape(1, -1)[0])
    #
    # for i in range(0,len(test_data)):
    #     rand_data = randrange(0,len(test_data))
    # print(prediction[rand_data])
    # print(test_labels.reshape(1, -1)[0][rand_data])
    # print(rand_data)
    #
    #
    # if prediction == 0:
    #     print("RED")
    # if prediction == 1:
    #     print("GREEN")
    # #
    # #
    # plt.imshow(test_images[rand_data, :, :], cmap='gray')
    # plt.title("Ground Truth : {} Prediction: {}".format(test_labels[rand_data], prediction))
    # plt.show()


# ------------------------------------------------------------------------------
# main starts from here
run_random_imges()