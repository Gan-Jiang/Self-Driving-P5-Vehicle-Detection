import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.externals import joblib
from scipy.misc import imresize
from produce_dataset import get_hog_features
import pandas as pd
import random

svc = joblib.load('svc.pkl')
X_scaler = joblib.load('X_scaler.pkl')

def get_car_obj():
    df = pd.read_csv("./object-dataset/labels.csv", header=None)
    image = df[[0]]
    image = image.as_matrix()
    car_dic = {}
    for ind, value in enumerate(image):
        temp = value[0].split(" ")
        if temp[6] == '"pedestrian"' or temp[6] == '"Street Lights"':
            continue
        try:
            #(xmin,ymin,xmax,ymax)
            car_dic[temp[0]].append((int(temp[1]),int(temp[2]),int(temp[3]),int(temp[4])))
        except:
            car_dic[temp[0]] = [(int(temp[1]),int(temp[2]),int(temp[3]),int(temp[4]))]
    return car_dic


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img_name, car_dic, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    global count
    img = mpimg.imread("./object-dataset/" + img_name)
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            image = img[starty:endy, startx:endx, :]

            image = imresize(image, (64, 64, 3))

            orient = 9
            pix_per_cell = 8
            cell_per_block = 2
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            hog_feature1 = get_hog_features(feature_image[:,:,0], orient,
                                        pix_per_cell, cell_per_block, feature_vec=True)
            hog_feature2 = get_hog_features(feature_image[:,:,1], orient,
                                            pix_per_cell, cell_per_block, feature_vec=True)
            hog_feature3 = get_hog_features(feature_image[:,:,2], orient,
                                            pix_per_cell, cell_per_block, feature_vec=True)
            features = np.concatenate((hog_feature1, hog_feature2, hog_feature3))
            features = features.reshape(1, -1)
            scaled_features = X_scaler.transform(features)
            prediction = svc.predict(scaled_features)

            # Append window position to list
            if prediction[0] == 1:
                judge = 1
                for i in car_dic[img_name]:
                    # (xmin,ymin,xmax,ymax)

                    if ((i[0] <= startx <= i[2]) and (i[1] <= starty <= i[3])) or ((i[0] <= endx <= i[2]) and (i[1] <= endy <= i[3])):
                        judge = 0
                        break
                if judge == 1:
                    #false positive
                    count+=1
                    mpimg.imsave("non-vehicles/extra/" + 'h'+ str(count) + '.png', image)

def get_car_obj2():
    df = pd.read_csv("./object-detection-crowdai/labels.csv", header=0)
    image = df
    image = image.as_matrix()
    for ind, value in enumerate(image):
        car_dic = {}
        for ind, value in enumerate(image):
            if value[5] == "Pedestrian":
                continue
            try:
                #(xmin,ymin,xmax,ymax)
                car_dic[value[4]].append((int(value[0]),int(value[1]),int(value[2]),int(value[3])))
            except:
                car_dic[value[4]] = [(int(value[0]),int(value[1]),int(value[2]),int(value[3]))]
        return car_dic


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window2(img_name, car_dic, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    global count
    img = mpimg.imread("./object-detection-crowdai/" + img_name)
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            image = img[starty:endy, startx:endx, :]

            image = imresize(image, (64, 64, 3))

            orient = 9
            pix_per_cell = 8
            cell_per_block = 2
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            hog_feature1 = get_hog_features(feature_image[:,:,0], orient,
                                        pix_per_cell, cell_per_block, feature_vec=True)
            hog_feature2 = get_hog_features(feature_image[:,:,1], orient,
                                            pix_per_cell, cell_per_block, feature_vec=True)
            hog_feature3 = get_hog_features(feature_image[:,:,2], orient,
                                            pix_per_cell, cell_per_block, feature_vec=True)
            features = np.concatenate((hog_feature1, hog_feature2, hog_feature3))
            features = features.reshape(1, -1)
            scaled_features = X_scaler.transform(features)
            prediction = svc.predict(scaled_features)

            # Append window position to list
            if prediction[0] == 1:
                judge = 1
                for i in car_dic[img_name]:
                    # (xmin,ymin,xmax,ymax)

                    if ((i[0] <= startx <= i[2]) and (i[1] <= starty <= i[3])) or ((i[0] <= endx <= i[2]) and (i[1] <= endy <= i[3])):
                        judge = 0
                        break
                if judge == 1:
                    #false positive
                    count+=1
                    mpimg.imsave("./non-vehicles/extra/" + 'i'+ str(count) + '.png', image)

def find_hard_negative(car_dic,car_dic2):
    global count
    for ind,i in enumerate(car_dic.keys()):
        rn = random.random()
        if rn < 0.33:
            slide_window(i, car_dic, x_start_stop=[None, None], y_start_stop=[300, None],
                                   xy_window=(270, 215), xy_overlap=(0.45, 0.45))
        elif rn > 0.66:
            slide_window(i, car_dic, x_start_stop=[None, None], y_start_stop=[300, None],
                         xy_window=(195, 180), xy_overlap=(0.45, 0.45))
        else:
            slide_window(i, car_dic, x_start_stop=[None, None], y_start_stop=[300, None],
                         xy_window=(165, 150), xy_overlap=(0.45, 0.45))
        print(count)
        print('ind'+str(ind))
        if count >= 5000:
            break
    count = 0
    for ind, i in enumerate(car_dic2.keys()):
        rn = random.random()
        if rn < 0.33:
            slide_window2(i, car_dic2, x_start_stop=[None, None], y_start_stop=[300, None],
                          xy_window=(270, 215), xy_overlap=(0.48, 0.48))
        elif rn > 0.66:
            slide_window2(i, car_dic2, x_start_stop=[None, None], y_start_stop=[300, None],
                          xy_window=(195, 180), xy_overlap=(0.48, 0.48))
        else:
            slide_window2(i, car_dic2, x_start_stop=[None, None], y_start_stop=[300, None],
                          xy_window=(165, 150), xy_overlap=(0.48, 0.48))
        print(count)
        print('ind'+str(ind))
        if count >= 5000:
            break

count = 0
car_dic = get_car_obj()
car_dic2 = get_car_obj2()
find_hard_negative(car_dic, car_dic2)