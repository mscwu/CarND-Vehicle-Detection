**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Normalize your features and randomize a selection for training and testing.
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./output_images/example_detection.jpg
[image5]: ./output_images/pipeline_example.jpg
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

In the end, the last combinations gave me the best accuracy. The final accuracy with HSV color space, 3 HOG channels, 32 spatial size and histogram bins, 9 orientaions, 8 pixcels per cell and 2 cells per block produced an accruacy of 98.97% on the test data set.  

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search is one of the most important and hardest part of the project. The concept is easy to understand but the real implementation requires a lot of fine tuning.  

The first step is to propose a multi scale window approach on different regions. We need to reach a balance between robust detection, execution time, false positives. Here is what I have found throughout the project:  

* Vehicles closer to the camera appears larger, therefore, larger windows should be used. However, these large windows provide limited detections without having a very high overlapping.
* Vehicles far away from the camera apperas very small. Smaller sized window are used. However, as the windows get smaller, the total number of windows increases quickly thus slowing down the detection pipeline.
* When the total number of windows is low, the pipeline runs fast. However, it makes it hard to distinguish false positve from true detection. Therefore, it is necessary to maintain a certain density of windows. The advantage is that, as the number of windows grows, we detect increasingly more vehicles than false positives. Later, we can apply techniques to reduce the number of false positives.
* For vehicles close to the camera, even a small window can extract part of the feature so it is not necessry to limit small windows only to the top part of the region of interest.  
* A the sliding window slides with a fixed step, and different windows sizes have different chances of introducing false positives, it is helpful to tune step size with the size of window.

Based on the above findings, I used the following sliding window scheme:  

* ystart, ystop, scale = [380, 700, 4]. This is mainly used to find vehicles close to the camera.
* ystart, ystop, scale = [370, 600, 2]. This is used to cover a majority of the image.
* ystart, ystop, scale = [360, 500, 1]. This is the smallest windows and is used to detect vehicles further down the road and part of vehicles a bit closer to the camera.

Here is an example of the windows.  

![alt text][image3]

After windows are proposed, HOG sub-sampling is implemented. The idea is that we only compute HOG once for each image and then use the sliding windows to examine part of the image. HOG is one of the operations in the pipeline that has the most computation time. By having it done only once for each frame, pipeline efficiency is increased.  

For each sub images contained in a sliding window, the HOG feature, spatial binning and histogram features are all fed through the linear SVM classifier to make a prediction. If the window content is predicted as a car, that window is added to a list to winows for further processing.  


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here is an example of the returned boxes. It can be told that the correction detection is far more than false positives, which lays a good fundation for later post-processing.

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)
Here's a [link on YouTube](https://youtu.be/tqexeLE131I)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.


I recorded the positions of positive detections in each frame of the video.  

Then an initial heat map was created based on the detection in the current frame. Then a threshold was applied to filter obvious false positives.  

The heat map is then smoothed using a Gaussian blur with a kernal size of 11. After Gaussian blur, a grey opening operation from `scipy.ndimage.morphology` was applied. This reduced the local maxima in the heat map that tended to affect later operations.  

After local maxima was removed, the mean and standard deviation of the nonzero elements in the image were calculated, and they were used to construct a second threshold.  

Apply this threshold to the heat map again to remove less obvioud false positives in the image.  The last step was to dilate the detected boxes as previous operations tended to shrink the area of detected window.  

Above steps were realised in the code as follows:  

```python
bboxes = multi_scale_search(road_image, multi_scale_windows, svc_hsv, X_scaler_hsv, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
heat = np.zeros_like(road_image[:,:,0]).astype(np.float)
heat = add_heat(heat, bboxes)
heat = apply_threshold(heat,7)
heat = cv2.GaussianBlur(heat,(11, 11),0)
heat = grey_opening(heat, size=(5,5))
mean = np.mean(heat[np.nonzero(heat)])
std = np.std(heat[np.nonzero(heat)])
threshold = mean - 2.5 * std
heat = apply_threshold(heat, threshold)
heat = grey_dilation(heat, size=(49,49))
```

Up till now, false positive reduction has been done on a per frame basis. However, in a frame stream, we can take advantage of history heat maps and use them to reduce the false positives even more.  

I constructed a class called `Heatmap()`.  

```python
class Heatmap():
    def __init__(self):
        # are there any bounding boxes found in the last iteration?
        self.detected = False
        # recent heat maps, hold last 5 frames
        self.recent_heat = []
        # best estimation of heatmap
        self.best_heat = []
        # current heamap
        self.current_heat = []
    
    def update(self, heatmap):
        self.detected = True
        self.current_heat = heatmap
        if len(self.recent_heat) == 5:
            # delete earliest heat map
            self.recent_heat.pop(0)
        self.recent_heat.append(self.current_heat)
        self.best_heat = np.sum(self.recent_heat, axis=0)
```

In this class, last 5 heat maps are recorded and an average of those are used as a best estimate of current heatmap.  

Then, a similar thresholding scheme was used on the best estimate, but with much higher confidence (higher threshold).  

This two step thresholding greatly reduced false positives and also smoothed the video output.  

Finally, to construct bounding boxes, `scipy.ndimage.measurements.label()` was used to find groups of objects.

Together with lane detection pipeline from last project, here is my final pipeline:  

```python
def final_pipeline(image):
    road_image = np.copy(image)
    lane_output = process_lane_image(road_image)
    undist_road_image = cal_undistort(road_image, objpoints, imgpoints)
    bboxes = multi_scale_search(road_image, multi_scale_windows, svc_hsv, X_scaler_hsv, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    heat = np.zeros_like(road_image[:,:,0]).astype(np.float)
    heat = add_heat(heat, bboxes)
    heat = apply_threshold(heat,7)
    heat = cv2.GaussianBlur(heat,(11, 11),0)
    heat = grey_opening(heat, size=(5,5))
    mean = np.mean(heat[np.nonzero(heat)])
    std = np.std(heat[np.nonzero(heat)])
    threshold = mean - 2.5 * std
    heat = apply_threshold(heat, threshold)
    heat = grey_dilation(heat, size=(49,49))
    heatmap.update(heat)
    heat_estimate = heatmap.best_heat
    heat_estimate_mean = np.mean(heat_estimate[np.nonzero(heat_estimate)])
    heat_estimate_std = np.std(heat_estimate[np.nonzero(heat_estimate)])
    threshold = heat_estimate_mean - 0.5 * heat_estimate_std
    final_heat = apply_threshold(heat_estimate, threshold)
    heatmap_clip = np.clip(final_heat, 0, 255)
    labels = label(heatmap_clip)
    draw_img = draw_labeled_bboxes(np.copy(lane_output), labels)
    return draw_img
```
Here's an example result showing the example image and its heatmap best estimation from `Heat()` object.  

![alt text][image5]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

* I started with spending a lot of time fine tuning the sliding window, without using a more complex two step thresholding a heat map history. It turned out that it was very hard to come up with a good solution of multi scale windows that not only detected all the vehicles but also had very low false positives. Later I borrowed the idea from last project and created the heat map history that really freed me from this deadend. With a heat map history and two step thresholding, I was able to focus on selecting a sliding window scheme that detected vehicles as mush as possible and then, use thresholding and history heat map to deal with false positives. The idea was that the false positives had to be persistent in a series of frames to be really considered in the final heat map estimation. However, the classifier was robust enough that even though false positives existed, they did not existed in a persistent manner. The video has 25 fps so 5 frames are a fraction of a second. The vehicle was not able to change its location drastically within 5 frames, which allow heat to accumulate around a vehicle while small changes in the images made false positives in images show up in different places thus they had no chance to accumulate like true vehicles.  
* If I had time I would create a larger training dataset. Current dataset includes about 8500 images but larger set would be helpful for the classifier to not produce false positives in the first place.
* I did not use hard negative mining to boot the performance of the classifier. This classification problem is "imbalanced" in nature as there are far more negatives in an image than the positives, i.e., vehicles. It might be a good idea to harvest the false positives and use them to retrain the classifier.
* It is possible to implement a "quick search" method just like the one from last project. For each frame, we can search for new vehicles while make focused effort in areas where vehicles have been deteced. This will enhance the tracking ability of the pipeline and also produce smoother results. On the otherhand, the two step thresholding and heat map history averaging may need to be changed according to work with the new detection scheme, such as changing the threshold, kernel size and confidence interval.