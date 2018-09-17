#import Packages
import numpy as np
import pandas as pd
from os import sep
from PIL import Image,ImageDraw
from random import randrange

#function to create random image
def fn_make_img(count=100, path="E:\Machine Learning\color_sep_data"):
    #check if file previously present, if yes then delete it and recreate
    #creating file
    with open(path + sep + "data.csv", mode="w+") as f:
        f.writelines("Image_Name,Flag\n")

        #save image in csv file, this will be the lookup table for training and test of data
    with open(path + sep + "data.csv", mode="a") as f:

            #take loop to generate random images
        for i in range(0, count):
            rand_call = randrange(0, 2)
            # print(i)
            # 0 = red
            # 1 = Green
            # if rand_call==0:
            #    rand_call="Red"
            # if rand_call==1:
            #    rand_call="Green"
            # print(rand_call)

            #save the call for correlating image name in a csv file
            #this will be the lookup table and also to create training and test data

            f.writelines(str(i) + ".png," + str(rand_call) + "\n")

            #generate the image

            new_image = Image.new('RGB',(32,32),color='white')

            #create rectangular border
            draw = ImageDraw.Draw(new_image)
            draw.rectangle(((0,0),(31,31)),fill='white',outline='black')

            #draw a circle inside
            #color based on rand call

            if rand_call == 0:
                draw.ellipse([2,2,28,28],outline='black',fill='red')
            if rand_call == 1:
                draw.ellipse([2,2,28,28],outline='black',fill='Green')

            #delete draw object
            del draw

            #save image
            new_image.save(path+sep+str(i)+".png")
    return

def split(df,headSize):
    hd = df.head(headSize)
    tl = df.tail(len(df)-headSize)
    return hd,tl

def load_data(img_path = "E:\Machine Learning\color_sep_data", path="E:\Machine Learning\color_sep_data" + sep +"data.csv", split_ratio=0.8,total_row=0):

    if total_row>0:
        data = pd.read_csv(path,sep=',',nrows=total_row)
    else:
        data = pd.read_csv(path,sep=',')

    #print(data.shape[0])
    #print(data.head())

    #split the data
    training_data_tracker, test_data_tracker = split(data,int(data.shape[0]*split_ratio))
    print(training_data_tracker)
    print(test_data_tracker)

    #once the split is done we loop through the training and test set,
    # and for each image we need to open the image and covert it to multidimensional array
    # where the array structure should be width, height, channel information
    # so shape would be [32,32,3]

    # and there is another dimension to be consider i.e. number of images
    # so we will create a placeholder for all images and then put the data
    # in the placeholder as when generated

    # in order to build placeholder dynamic we take width and height info

    op_image = Image.open(img_path + sep + data.ix[0][0])
    (width , height) = op_image.size

    #also we are making the channel info more dynamic
    #monochrome grayscale
    if len(np.array(op_image).shape) == 2:
        channel = 1

    #color image
    else:
        channel = np.array(op_image).shape[2]

    # now lets take target part separately

    y_train = training_data_tracker['Flag'].values.reshape(-1,1)
    y_test = test_data_tracker['Flag'].values.reshape(-1,1)


    # now lets take the feature/training part
    # create a default placeholder for <no. of images>,<width>,<height>,<channel>

    # for training data
    x_train = np.zeros((training_data_tracker.shape[0], width, height, channel),dtype='uint8')

    for i in training_data_tracker.index:
        print(training_data_tracker.ix[i][0])
        #imgname = data.iloc[i,]['Image_Name']  # another way to get the image name
        #print(imgname)
        op_image = Image.open(img_path +sep+ training_data_tracker.ix[i][0])
        (width,height) = op_image.size

        op_image_map = list(op_image.getdata())
        op_image_map = np.array(op_image_map)
        x_train[i,:,:,:] = op_image_map.reshape(width,height,channel)

        # for test data
    x_test = np.zeros((test_data_tracker.shape[0], width, height, channel), dtype='uint8')

    test_data_tracker = test_data_tracker.reset_index(drop=True)
    for i in test_data_tracker.index:
        print(test_data_tracker.ix[i][0])
        # imgname = data.iloc[i,]['Image_Name'] # another way to get the image name
        op_img = Image.open(img_path + sep + test_data_tracker.ix[i][0])
        (width, height) = op_img.size
        op_img_map = list(op_img.getdata())
        op_img_map = np.array(op_img_map)
        x_test[i, :, :, :] = op_img_map.reshape(width, height, channel)
    return x_train,y_train,x_test,y_test




#--------------------------------------------------------------
# main
# fn_make_img(count=100)
load_data(total_row=100)
