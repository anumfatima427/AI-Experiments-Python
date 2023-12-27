# -*- coding: utf-8 -*-
"""
Gabor Filter is a band pass filter
used mainly for edge detection, texture analysis and feature extraction

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd

img = cv2.imread(r"D:\Anum\Learning ML DL\6-Random Forest for Image Segmentation\plantcell.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#create an empty dataframe and start adding features to it
df = pd.DataFrame()
img2 = img.reshape(-1)
df['Original_values'] = img2
#up until now we have created a dataframe and added original pixel values to it

num = 1
for theta in range(2):
    theta = theta /4. *np.pi
    for sigma in (3, 5):
        for lambd in np.arange(0, np.pi, np.pi/4.):
            for gamma in (0.05, 0.5):
                #lets define gabor labels, these will be column names for different gabor features we are generating
                gabor_label = 'Gabor' + str(num)
                #after identifying the values, we want to generate a kernel for each value!
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)  
                #once the kernel is generated, apply it on the original image
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                #add these as a new column to the dataframe
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img
                num += 1
                print(gabor_label)


df.to_csv(r"D:\Anum\Learning ML DL\6-Random Forest for Image Segmentation\Gabor_features.csv")
print(df.head())
