#This is used to get extra data.
import numpy as np
import glob
import matplotlib.image as mpimg
import pickle
from scipy.misc import imresize
import pandas as pd

#define a function to get cars image from udacity dataset
def get_extra_cars():
    df = pd.read_csv("./object-dataset/labels.csv", header=None)
    image = df[[0]]
    image = image.as_matrix()

    for ind, value in enumerate(image):
        print(ind)
        temp = value[0].split(" ")
        if temp[6] == '"car"':
            temp_image = mpimg.imread("./object-dataset/" + temp[0])

            temp_image = temp_image[int(temp[2]):int(temp[4]) + 1, int(temp[1]):int(temp[3]) + 1, :]
            image = imresize(temp_image, (64, 64, 3))
            mpimg.imsave("object-dataset/result/"+str(ind)+'.png', image)

    df = pd.read_csv("./object-detection-crowdai/labels.csv", header=0)
    image = df
    image = image.as_matrix()

    for ind, value in enumerate(image):
        print(ind)
        if value[5] == "Car":
            temp_image = mpimg.imread("./object-detection-crowdai/" + value[4])
            temp_image = temp_image[int(value[1]):int(value[3]) + 1, int(value[0]):int(value[2]) + 1, :]

            image = imresize(temp_image, (64, 64, 3))
            mpimg.imsave("object-detection-crowdai/result/"+str(ind)+'.png', image)

#get_extra_cars()
