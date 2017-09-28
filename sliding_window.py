#This file is used for implementing sliding windows and drawing box.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.externals import joblib
from scipy.misc import imresize
from produce_dataset import get_hog_features
from skimage.feature import blob_doh
from scipy import ndimage as ndi
from skimage.morphology import watershed

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Iterate through the bounding boxes.
    img2 = img.copy()
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(img2, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return img2


def produce_heatmap(prediction, position, scale):
    map = np.zeros([720, 1280])
    for i in range(len(prediction)):

        if prediction[i] > 0:
            map[int(360 + position[i][1] * 8 * scale[1]):int(360 + position[i][3] * 8 * scale[1]), int(position[i][0] * 8 * scale[0]):int(position[i][2] * 8 * scale[0])]+=min(10, prediction[i])
    return map

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, svc, X_scaler, x_start_stop=[0, None], y_start_stop=[0, None],
                    xy_window=(7, 7), xy_overlap=(0.8, 0.8), scale = (1, 1)):
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # convert to YcrCb
    image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    image = image[360:, :, :]    #shape 360, 1280, 3

    image = imresize(image, (int(360/scale[1]), int(1280/scale[0]), 3))

    hog_1 = get_hog_features(image[:, :, 0], orient,
                                    pix_per_cell, cell_per_block, feature_vec=False)

    hog_2 = get_hog_features(image[:, :, 1], orient,
                                    pix_per_cell, cell_per_block, feature_vec=False)

    hog_3 = get_hog_features(image[:, :, 2], orient,
                                    pix_per_cell, cell_per_block, feature_vec=False)


    x_start_stop[1] = hog_1.shape[1]
    if y_start_stop[1] == None:
        y_start_stop[1] = hog_1.shape[0]

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
    sample_list = []
    position = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            if endy > hog_1.shape[0] or endx > hog_1.shape[1]:
                continue
            hog_feature1 = hog_1[starty:endy, startx:endx:, :, :, :].ravel()
            hog_feature2 = hog_2[starty:endy, startx:endx:, :, :, :].ravel()
            hog_feature3 = hog_3[starty:endy, startx:endx:, :, :, :].ravel()

            features = np.concatenate((hog_feature1, hog_feature2, hog_feature3))
            features = features.reshape(1, -1)
            scaled_features = X_scaler.transform(features)

            sample_list.append(scaled_features.reshape(5292))
            position.append([startx, starty, endx, endy])
            #use a threshold for prediction
    prediction = svc.decision_function(sample_list)

    return produce_heatmap(prediction, position, scale)


def process_image(file):
    svc = joblib.load('svc.pkl')
    X_scaler = joblib.load('X_scaler.pkl')
    image = cv2.imread('test_images/' + file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    windows1 = slide_window(image, svc, X_scaler, x_start_stop=[0, None], y_start_stop=[0, 30],
                            xy_window=(7, 7), xy_overlap=(0.8, 0.8), scale = (1.1, 0.8))

    windows2 = slide_window(image, svc, X_scaler, x_start_stop=[0, None], y_start_stop=[0, 14],
                            xy_window=(7, 7), xy_overlap=(0.8, 0.8), scale=(2, 1.7))

    windows3 = slide_window(image, svc, X_scaler, x_start_stop=[0, None], y_start_stop=[0, None],
                            xy_window=(7, 7), xy_overlap=(0.8, 0.8), scale=(3, 2))

    windows4 = slide_window(image, svc, X_scaler, x_start_stop=[0, None], y_start_stop=[0, 17],
                            xy_window=(7, 7), xy_overlap=(0.8, 0.8), scale=(1.7, 1.4))

    windows5 = slide_window(image, svc, X_scaler, x_start_stop=[0, None], y_start_stop=[0, 10],
                            xy_window=(7, 7), xy_overlap=(0.8, 0.8), scale=(1.4, 1.1))
    heat = windows1 +  windows2 + windows5  + windows3 + windows4

    heat = apply_threshold(heat, min(heat.max()/3 + heat.std()*4,15))

    blobs_doh = blob_doh(heat, min_sigma = 1, max_sigma=50, threshold=.01)

    distance = ndi.distance_transform_edt(heat)
    markers = np.zeros([image.shape[0], image.shape[1]])
    count = 1
    for i in blobs_doh[:, :2]:
        markers[int(i[0]), int(i[1])] = count
        count+=1

    labels = watershed(-distance, markers, mask=heat)
    labels = labels.astype('int')


    window_list = []


    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(heat.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        c = cv2.boundingRect(c)
        window_list.append(((c[0], c[1]), (c[0]+c[2], c[1]+c[3])))

    window_img = draw_boxes(image, window_list, color=(0, 0, 255), thick=6)

    fig, axes = plt.subplots(nrows = 2, ncols=2, figsize=(15, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    ax0 = axes[0, 0]
    ax1 = axes[0, 1]
    ax2 = axes[1, 0]
    ax3 = axes[1, 1]

    ax0.imshow(image,  interpolation='nearest')
    ax0.set_title('Original image')
    ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
    ax1.set_title('Distances')
    ax2.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
    ax2.set_title('Separated objects')
    ax3.imshow(window_img)
    ax3.set_title('Final image')

    plt.imshow(window_img)

process_image('test3.jpg')