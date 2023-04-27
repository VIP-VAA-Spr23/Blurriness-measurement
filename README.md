
# Images Blurriness Measurement and IOU Score Calculation

This repository aims to perform image blurriness on a set of images, and calculates their Intersection Over Union (IOU) scores with the corresponding ground truth bonding boxes, then visualizes the results using a scatter plot and a histogram. The results is exported to an Excel file for futher analysis.


## Features

- Reads images and corresponding XML files containing ground truth bounding box information.
- Crops images based on the bounding box coordinates.
- Resizes cropped images using seam carving to maintain aspect ratio.
- Calculates the Laplacian variance of each image as a measure of blurriness.
- Calculates IOU scores between the predicted bounding boxes and the ground truth bounding boxes.
- Performs linear regression on the blurriness and IOU scores to visualize the relationship.
- Visualizes the distribution of blurriness for true positive and false negative/miss detections using a histogram.
- Exports the results to an Excel file.

## Imported Libraries
- numpy
- opencv-python (cv2)
- glob
- os
- pandas
- xml.etree.ElementTree (ET)
- PIL (Image)
- skimage (io, transform)
- torch
- sklearn (LinearRegression)
- matplotlib (pyplot)
- json
## How to Use the Project

1. Clone the repository.
2. Organize your data:
   - Place all the images in a single folder.
   - Label your images using the [LabelImg tool](https://github.com/pranjalAI/labelImg) or a similar tool. Place all the XML files containing the ground truth labels in another folder.
   - Create two separate folders: one for saving cropped images and another for saving resized images.
3. Update the paths in the `IOU_score_calculation.py` and `Bluriness_detection.py` script for the following:
   - Images folder
   - XML files folder
   - Cropped images folder
   - Resized images folder
4. Run the script using `python image_analysis.py`.
5. The script will generate scatter plots, histograms, and an Excel file with the results.
