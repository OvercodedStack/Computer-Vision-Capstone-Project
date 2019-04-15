################################################################
#  Computer vision final project
#  Medical Lung Detection Program
#  Written by Esteban Segarra Martinez
#  OpenCV programme
#
################################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pylab import arange,array,uint8
import math
import scipy as sy


img = cv2.imread('c_1.jpeg')
#img = cv2.imread('c_3.png')
############CROP IMAGE TO A DIMENSION AND GRAYSCALE##############
height = img.shape[0]
width  = img.shape[1]
l_marH = int(height*.05)
r_marH = int(height*.85)
l_marW = int(width*.1)
r_marW = int(width*.9)
img = img[l_marH:r_marH, l_marW:r_marW]
size = img.shape[0] * img.shape[0]
graded_size = int(size *.008)
print(size)
print(graded_size)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

##############ADJUST CONTRAST#####################################
maxIntensity = 255.0 # depends on dtype of image data
x = arange(maxIntensity)

########### Parameters for manipulating image data################
phi = 1
theta = .4
################################################
gray = (maxIntensity/phi)*(gray/(maxIntensity/theta))**2
gray = array(gray,dtype=uint8)


###########THRESHOLD AND SHARPEN + EQUALIZE########################
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel_sharpening = np.array([[-1,-1,-1],
                              [-1, 11,-1],
                              [-1,-1,-1]])
gray = cv2.filter2D(gray, -1, kernel_sharpening)
gray = cv2.equalizeHist(gray)

##############################FIND CONTOURS OF THE LUNGS###############
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
__,contours,__ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.imshow("gray",gray)

######################CROP OUT THE LUNG IMAGES##############################
out = np.zeros((height, width, 3), np.uint8)
list_contours = []
counter = 0
for item in range(0,len(contours)):
    if contours[item].size > graded_size:
        list_contours[counter] = contours[item]
        cv2.drawContours(img, contours, item, (0, 0, 255), 1)
        #print("ping", contours[item].size, item )
        mask = np.zeros_like(gray)  # Create mask where white is what we want, black otherwise
        cv2.drawContours(mask, contours, item, 255, -1)  # Draw filled contour in mask
        out = np.zeros_like(gray)  # Extract out the object and place into output image
        out[mask == 255] = gray[mask == 255]

        # Now crop
        x, y = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        out = out[topx:bottomx + 1, topy:bottomy + 1]
        counter += 1

###################BLOB DETECTION PHASE #####################################
out = cv2.GaussianBlur(out,(11,11),0)


################################################### Sift####################
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(out, None)

#
# for point in range(0, len(kp)):
#     for point_2 in range(0,len(list_contours)):
#         if point > list_contours[point_2]:


print(type(kp))
img = cv2.drawKeypoints(out, kp, img)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
############################################################################


def calc_dist(point_1, point_2):
    return math




#img = cv2.imread('c_2.jpeg',cv2.IMREAD_COLOR)
#
# surf = surf = cv2.xfeatures2d.SURF_create(400)
# surf.setHessianThreshold(2500)
# kp, des = surf.detectAndCompute(out,None)
# img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
# plt.imshow(img2),plt.show()
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#

