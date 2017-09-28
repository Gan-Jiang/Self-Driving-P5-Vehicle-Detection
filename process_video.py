from sliding_window import slide_window, draw_boxes, apply_threshold
from moviepy.editor import VideoFileClip
from skimage.feature import blob_doh
from scipy import ndimage as ndi
from skimage.morphology import watershed
from sklearn.externals import joblib
import numpy as np
import cv2
from scipy.misc import imsave
#vehicle class to record the recent vehicles detection.
class Vehicle():
    def __init__(self):
        #each vehicle is represented by [startx, starty, endx, endy, count1, count2]
        self.vehicle_list = []

    #compute the distance between two points
    def compute_distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))


    def update_list(self, find_boxes):
        for i in range(len(self.vehicle_list)):
            self.vehicle_list[i][5] += 1
        for i in find_boxes:
            existed = False
            startx, starty = i[0]
            endx, endy = i[1]
            for ind, j in enumerate(self.vehicle_list):
                if self.compute_distance(np.array([(startx + endx)/2, (starty + endy)/2]), np.array([(j[0] + j[2])/2, (j[1] + j[3])/2])) <= 28:
                    self.vehicle_list[ind] = [(startx+j[0])//2, (starty+j[1])//2, (endx+j[2])//2, (endy+j[3])//2, self.vehicle_list[ind][4]+1, max(self.vehicle_list[ind][5]-2,0)]
                    existed = True
                    break
            if existed == False:
                self.vehicle_list.append([startx, starty, endx, endy, 1, 0])


    def return_list(self):
        windows_list = []
        del_list = []
        for ind, i in enumerate(self.vehicle_list):
            if i[5] >= 15:
                del_list.append(ind)
            if i[4] >= 25:
                windows_list.append(((i[0], i[1]), (i[2], i[3])))
        self.vehicle_list = [i for j, i in enumerate(self.vehicle_list) if j not in del_list]
        return windows_list

# Malisiewicz et al.
def non_max_suppression_fast(window_list, overlapThresh):
    # if there are no boxes, return an empty list
    if len(window_list) <= 1:
        return window_list
    boxes = np.zeros([len(window_list), 4])
    for i in range(len(window_list)):
        boxes[i,:] = [window_list[i][0][0], window_list[i][0][1], window_list[i][1][0], window_list[i][1][1]]
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    window_list = [window_list[index] for index in pick]
    return window_list




vehicle = Vehicle()
frame = 0
svc = joblib.load('svc.pkl')
X_scaler = joblib.load('X_scaler.pkl')
def process_image(image):
    global vehicle, frame, svc, X_scaler
    frame += 1

    if frame < 126:
        return image

    windows1 = slide_window(image, svc, X_scaler, x_start_stop=[0, None], y_start_stop=[0, 30],
                                xy_window=(7, 7), xy_overlap=(0.8, 0.8), scale=(1.1, 0.8))

    windows2 = slide_window(image, svc, X_scaler, x_start_stop=[0, None], y_start_stop=[0, 14],
                            xy_window=(7, 7), xy_overlap=(0.8, 0.8), scale=(2, 1.7))

    windows3 = slide_window(image, svc, X_scaler, x_start_stop=[0, None], y_start_stop=[0, None],
                            xy_window=(7, 7), xy_overlap=(0.8, 0.8), scale=(3, 2))

    windows4 = slide_window(image, svc, X_scaler, x_start_stop=[0, None], y_start_stop=[0, 17],
                            xy_window=(7, 7), xy_overlap=(0.8, 0.8), scale=(1.7, 1.4))

    windows5 = slide_window(image, svc, X_scaler, x_start_stop=[0, None], y_start_stop=[0, 10],
                            xy_window=(7, 7), xy_overlap=(0.8, 0.8), scale=(1.4, 1.1))
    heat = windows1 + windows2 + windows5 + windows3 + windows4

    heat = apply_threshold(heat, min(heat.max() / 3 + heat.std() * 4, 15))
    #find the blobs
    if heat.max() == 0:
        vehicle.update_list([])
        window_list = vehicle.return_list()
        window_list = non_max_suppression_fast(window_list, 0.2)
        window_img = draw_boxes(image, window_list, color=(0, 0, 255), thick=6)
        return window_img

    blobs_doh = blob_doh(heat, min_sigma=1, max_sigma=50, threshold=.01)
    if len(blobs_doh) == 0:
        vehicle.update_list([])
        window_list = vehicle.return_list()
        window_list = non_max_suppression_fast(window_list, 0.2)
        window_img = draw_boxes(image, window_list, color=(0, 0, 255), thick=6)
        return window_img

    #use watershed to find separated objects
    distance = ndi.distance_transform_edt(heat)
    markers = np.zeros([image.shape[0], image.shape[1]])

    count = 1
    for i in blobs_doh[:, :2]:
        markers[i[0], i[1]] = count
        count += 1

    labels = watershed(-distance, markers, mask = heat)
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
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        c = cv2.boundingRect(c)
        window_list.append(((c[0], c[1]), (c[0] + c[2], c[1] + c[3])))

    #compare with recent detections.
    vehicle.update_list(window_list)
    window_list = vehicle.return_list()
    window_list = non_max_suppression_fast(window_list, 0.2)

    window_img = draw_boxes(image, window_list, color=(0, 0, 255), thick=6)
    return window_img

white_output = 'test8.mp4'
clip1 = VideoFileClip("./project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)