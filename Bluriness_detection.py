import numpy as np
import cv2
import glob
import os
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
from skimage import io, transform
import torch
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import json
from IOU_score_calculation import calculate_iou_scores_v2
from IOU_score_calculation import get_image_sizes

# update the path below with the path of images.
Images_path = "C:/Users/17654/Desktop/Images"


# update the path below with the path of the labeled XML files.
XML_Folder_path = "C:/Users/17654/Desktop/labeled files"

# Set up the destination directory where you intend to store the cropped images,
# and update the path below with the path of the newly created folder.
Cropped_Img_Path = "C:/Users/17654/Desktop/Cropped Image"

# Set up the destination directory where you intend to store the resized images,
# and update the path below with the path of the newly created folder.
Resized_Img_path = "C:/Users/17654/Desktop/resized img"


def read_xml():
    # read xml files and extract the coodinates of each bonding boxes
    # and return a dictionary with keys correspond to image name and
    # values correspnd to coodinates

    dict = {}
    count = 0
    for file in os.listdir(XML_Folder_path):
        if file.endswith('.xml'):
            tree = ET.parse(os.path.join(XML_Folder_path,file))
            root = tree.getroot()
            temp = []
            filename = os.path.splitext(file)[0]+'.jpg'
            for object in root.findall('object'):
                n = object.find('bndbox')
                xmin = float(n.find('xmin').text)
                xmax = float(n.find('xmax').text)
                ymin = float(n.find('ymin').text)
                ymax = float(n.find('ymax').text)
                temp.append((xmin, ymin, xmax, ymax))
                count+=1
            dict[filename] = temp
    return dict

def crop_img(coordinates, img_path, new_dir):

    # This function takes in coodinates, image path and a new directory that
    # images are saved and crop all the bonding boxes in each image according
    # to the coodinates and save it to the new_dir.

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = Image.open(img_path)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for i, coords in enumerate(coordinates):
        cropped_img = img.crop(coords)
        if cropped_img.size[0] == 0 or cropped_img.size[1] == 0:
            print(f"Empty or invalid image for {img_name}_{coords}.jpg")
            continue
        cropped_img.save(os.path.join(new_dir, f"{img_name}_{coords}.jpg"))
    """
    for i, coords in enumerate(coordinates):
       cropped_img = img.crop(coords)
       cropped_img.save(os.path.join(new_dir, f"{img_name}_{coords}.jpg"))
    """
def save_crop_img(new_dir):

    # This function first call the read_xml() funciton to get the dictionary,
    # and then save each cropped image in new_dir
    temp = read_xml()

    for image_name, coords in temp.items():
        image_path = os.path.join(Images_path, image_name)
        crop_img(coords, image_path, new_dir)

#replace this with the new destination folder path you want to save
save_crop_img(Cropped_Img_Path)

def coords_mids():

    # This function reads the coordinates from the XML files using the read_xml()
    # function and calculates the average of all the coordinates. It returns the
    # calculated average as a list of four values.

    dict11 = read_xml()
    combined_coords = []
    for coords in dict11.values():
        for values in coords:
            combined_coords.append(values)
    average = [sum(x) / len(x) for x in zip(*combined_coords)]
    return average

def resize_img():

    # This function resizes the images to their average width and height
    # using seam carving, so that there're no external affect influencing
    # blurriness measurement

    path1 = glob.glob(Cropped_Img_Path + "/*.jpg")
    average = coords_mids()
    width = average[2] - average[0]
    height = average[3] - average[1]


    for img_path in path1:
        img = cv2.imread(img_path)
        resized_img = transform.resize(img, (height, width), preserve_range=True)
        filename = os.path.basename(img_path)
        io.imsave(f'{Resized_Img_path}/{filename}', resized_img.astype(img.dtype))

# Perform the resize function
resize_img()

def get_name():

    # A function that get the name of every name of file in
    # the desination folder

    file_name = os.listdir(Resized_Img_path)

    return file_name

def read_img():

    # A function that read every image in the destination folder
    # and return a list for future data processing

    path = glob.glob(Resized_Img_path + "/*.jpg")
    img_list = []
    for img in path:
        temp = cv2.imread(img)
        img_list.append(temp)

    return img_list

def laplacian_variance(path):

    # A function that calculate the laplacian variance of img
    # this is the main tool used for analysis bluriness of img
    lp_var = cv2.Laplacian(img, cv2.CV_64F).var()

    return lp_var

def IOU():

    # This function read the json file that stored the data in the localizer and return
    # a flat IOU score list
    image_sizes = get_image_sizes(Images_path)
    ground_truth = read_xml()
    json_file_path = "C:/Users/17654/PycharmProjects/Image analysis/venv/l_dict.json"
    iou_scores = calculate_iou_scores_v2(json_file_path, ground_truth, image_sizes)
    flat_IOU_list = [item for sublist in iou_scores for item in sublist]

    return (flat_IOU_list)


def linear_regression(blur_list, iou_scores):

    # This function takes in two lists: blur list, iou scores list and form a linear regression
    # model
    X = np.array(blur_list).reshape(-1, 1)
    y = np.array(iou_scores).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    blurred_points = [(x, y) for x, y, blur in zip(X, y, blur_list) if blur < 100]
    sharp_points = [(x, y) for x, y, blur in zip(X, y, blur_list) if blur >= 100]

    plt.scatter(*zip(*blurred_points), color='red', label='Blur')
    plt.scatter(*zip(*sharp_points), color='green', label='Sharp')

    plt.plot(X, y_pred, color='blue', label='Fitted line')
    plt.xlabel('Blur')
    plt.ylabel('IoU Score')
    plt.legend()
    plt.show()

def hist(list1, list2):
    histogram1 = []
    histogram2 = []
    for val1, val2 in zip(list1, list2):
        if val2 > 0.5:
            histogram1.append(val1)
        else:
            histogram2.append(val1)

    bin_width = 100
    plt.hist(histogram1, bins=range(int(min(histogram1)), int(max(histogram1)) + bin_width, bin_width), alpha=0.5, label='True Positive')
    plt.hist(histogram2, bins=range(int(min(histogram2)), int(max(histogram2)) + bin_width, bin_width), alpha=0.5, label='False Negative/Miss')

    plt.xlabel('Blurriness')
    plt.ylabel('Frequency')
    plt.legend()

    plt.show()

if __name__ == "__main__":

    # main function that display result
    # threshold set according to the median of variance of images
    i_list = read_img()
    blur_list = []
    mid = []
    for img in i_list:
        cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cmp = laplacian_variance(cvt)
        blur_list.append(cmp)
        threshold = 1178.5249339404377
        if cmp > threshold:
            dis = "Above mid"
        else:
            dis = "Below mid"
        mid.append(dis)
    IOU_Score = IOU()
    linear_regression(blur_list, IOU_Score)
    hist(blur_list, IOU_Score)


    median = np.median(blur_list)
    name = get_name()
    # set up dictionary which has keys with image name and values
    # are blurriness of image and whether or not it below or above median
    dictionary = dict(zip(name, zip(blur_list, mid)))
    print(dictionary)

    # set up excel to display result
    df = pd.DataFrame(list(dictionary.items()), columns=['img_name','blur value'])
    df.to_excel('blur detection.xlsx', index=False)







