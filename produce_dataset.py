#Used to get the training set and validation set.
import numpy as np
import glob
import cv2
import matplotlib.image as mpimg
import pickle
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.misc import imresize
from random import shuffle, sample
from sklearn.externals import joblib


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, feature_vec=True):
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                   visualise=False, feature_vector=feature_vec)
    return features

# Define a function to get all features. This is for producing the dataset
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256), orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        #image = imresize(image, (64,64,3))
        # apply color conversion if other than 'RGB'
        if cspace != 'BGR':
            if cspace =='YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            elif cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        else:
            feature_image = np.copy(image)
        # Call get_hog_features() with vis=False, feature_vec=True
        hog_features1 = get_hog_features(feature_image[:, :, 0], orient,
                                        pix_per_cell, cell_per_block, feature_vec=True)
        hog_features2 = get_hog_features(feature_image[:, :, 1], orient,
                                        pix_per_cell, cell_per_block, feature_vec=True)
        hog_features3 = get_hog_features(feature_image[:, :, 2], orient,
                                        pix_per_cell, cell_per_block, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(np.concatenate((hog_features1, hog_features2, hog_features3)))
    # Return list of feature vectors
    return features










'''
images = glob.glob('object-dataset/result/*.png')

for image in images:
    cars.append(image)

images = glob.glob('object-detection-crowdai/result/*.png')

for image in images:
    cars.append(image)

shuffle(cars)
cars = sample(cars, 5000)
'''
'''
cars = []
notcars = []
images = glob.glob('vehicles/GTI_Far/*.png')

for image in images:
    cars.append(image)


images = glob.glob('vehicles/GTI_Left/*.png')

for image in images:
    cars.append(image)


images = glob.glob('vehicles/GTI_MiddleClose/*.png')

for image in images:
    cars.append(image)


images = glob.glob('vehicles/GTI_Right/*.png')

for image in images:
    cars.append(image)


images = glob.glob('vehicles/KITTI_extracted/*.png')

for image in images:
    cars.append(image)


images = glob.glob('non-vehicles/Extras/*.png')

for image in images:
    notcars.append(image)

images = glob.glob('non-vehicles/GTI/*.png')

for image in images:
    notcars.append(image)

images = glob.glob('non-vehicles/extra/*.png')

for image in images:
    notcars.append(image)


orient = 9
pix_per_cell = 8
cell_per_block = 2

car_features = extract_features(cars, orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block, cspace='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), hog_channel = 0)

notcar_features = extract_features(notcars, orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block, cspace='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), hog_channel = 0)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

joblib.dump(X_scaler, 'X_scaler.pkl')

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
#X_train, X_test, y_train, y_test = train_test_split(
#    scaled_X, y, test_size=0.1, random_state=23)

X_train = scaled_X
y_train = y
X_test = []
y_test = []
data = {}
data['X_train'] = X_train
data['X_test'] = X_test
data['y_train'] = y_train
data['y_test'] = y_test

pickle.dump(data, open("data.p", "wb"))
'''