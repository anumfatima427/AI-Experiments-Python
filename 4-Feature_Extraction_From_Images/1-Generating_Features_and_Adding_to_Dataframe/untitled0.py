#Gabor Filter can be used for texture analysis, edge detection, feature extraction
#Band Pass Filter 

import numpy as np
import cv2
from matplotlib import pyplot as plt

ksize = 5 #pixel size
sigma = 3 #variation -standard deviation
theta = 1*np.pi/4 #rotated by 50 degrees
lambd = 1*np.pi/4
gamma = 0.5
phi = 0

kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, phi, ktype=cv2.CV_32F)  
plt.imshow(kernel)


#amazing in texture, you can use them on two different regions with similar gray levels
#you can see the levels, but cannot extract or separate them, use gabor!

#now let's apply this filter on a image

# Using raw string with an 'r' prefix
img = cv2.imread(r"D:\Anum\Learning ML DL\Segmentation_Using_ML\Generating_Features\Yeast.jpg")
plt.imshow(img)
cv2.imshow('Original Image', img)
cv2.waitKey()
cv2.destroyAllWindows()