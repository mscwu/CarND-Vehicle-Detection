**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Normalize your features and randomize a selection for training and testing.
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./outputimages/car_not_car.jpg
[image2]: ./outputimages/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.  

For a machine learning problem, it is very important to work with a good dataset from the very beginning. GTI and KITTI image databases are widely used for research and industry purposes. Additionaly, CrowdAI also provides label vehicle images gathered in the US. After observing the GTI database, I immediately found that the vehicle images are collected based on a continuous video stream therefore, even though there are over 3000 vehicle images in the database, to optimize the classifier, I only plan to use a fraction of them. After subsampling the GTI dataset, new data from CrowAI is added to balance the positive and negative samples. Details on the implementation can be found in this [notebook](https://github.com/mscwu/udacity_vehicle_detection/blob/master/create_pickled_data.ipynb).

After a good dataset is created, I stored it in a pickle file.  

To extract HOG features, I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces. Details can be found in this [notebook](https://github.com/mscwu/udacity_vehicle_detection/blob/master/Color_Space_Exploration.ipynb).  
I tried 5 color spaces in total. They are `RGB`, `HSV`, `HLS`, `YUV` and `LAB`. Out of the 5 color spaces, RGB and HSV seemed to standout. I then kept these two options and tested both of them when I built the classifier.  
Here is an example using the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I mainly tried different number of HOG channels. I started with only 1 HSV HOG channel and later switched to using all of the channels as that offered improvements in classifier accuracy. Other HOG parameters did offer observable improvements.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I built the following training routine:  
```python
def train_SVC(vehicle, non_vehicle, color_space='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL', 
              spatial_size=(32,32), hist_bins=32,
              spatial_feat=True, hist_feat=True, hog_feat=True):
    t=time.time()
    car_features = extract_features(vehicle, color_space=color_space, spatial_size=spatial_size,
                                    hist_bins=hist_bins, orient=orient, 
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(non_vehicle, color_space=color_space, spatial_size=spatial_size,
                                    hist_bins=hist_bins, orient=orient, 
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract features...')  
    
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    print('Color space', color_space, '\nOrient', orient,
          '\nPixels per cell', pix_per_cell, '\nCell per block', cell_per_block, '\nHOG channel', hog_channel,
          '\nSpatial size', spatial_size, '\nHistogram bins', hist_bins)
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    accuracy = round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', accuracy) 
    
    return svc, accuracy, X_scaler
```
The training routine takes in vehicle and non vehicle images sets, along with other parameter such as color space and HOG orientations. Then, HOG, spatial and hisgram features were extracted from each image and stacked together to form a feature vector. After that, they were normalized by using `StandardScalar()` from `sklearn`. The feature vectors and corresponding labels were then split up into randomized training and test sets, 80% of data were kept as training data. Last, a linear SVM classifier was used to training on the training data and the accuracy of the test data was reported.  
I tried the following combinations:  
| Combination     | Color Space   | # of HOG Channels  | Spatial Size | Histogram Bins |
| :-------------: |:-------------:| :-----------------:| :---------:  | :------------: |
| 1        | RGB | 1              |        16      |        16        |
| 2        | HSV | 1              |        16      |        16        |
| 3        | RGB | 3              |        16      |        16        |
| 4        | HSV | 3              |        16      |        16        |
| 5        | HSV | 3              |        32      |        32        |

In the end, the last combinations gave me the best accuracy. The final accuracy with HSV color space, 3 HOG channels, 32 spatial size and histogram bins, 9 orientaions, 8 pixcels per cell and 2 cells per block produced an accruacy of 98.98% on the test data set.
###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

