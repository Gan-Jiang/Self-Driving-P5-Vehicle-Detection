## Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, I write a software pipeline to detect vehicles in a video. 

The Project
---

The goals / steps of this project are the following:

(1) Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier.
(2) Implement a sliding-window technique and use the trained classifier to search for vehicles in images. First, I only slide the windows in the bottom half of the image because in the top half there will not be vehicles. As I want to use multiple scales of the windows, I then resize the image by the scale parameter. After scaling, I get all hog features in three channels. Then, I slide the window and use linearSVM to predict if the corresponding image is a vehicle or not.

After I see the six test images, I decide to use five scales which should be able to cover vehicles in different position of the image. The scales are (70, 50), (90, 70), (110, 90), (130, 110) and (200, 130). I rescale these scales to fit the shapes of hogfeatures. I try different overlap windows and decide at last to use (0.8, 0.8) because it has the best performance.
(3) Run the pipeline on a video stream. 

In report.ipynb, I also discussed what I did to deal with the overlapping bounding boxes and false positives. 

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  Some example images for testing your pipeline on single frames are located in the `test_images` folder.  
