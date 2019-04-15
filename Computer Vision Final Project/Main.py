import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy as sy


img = cv2.imread('c_1.jpeg')
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# __,contours,__ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
# for item in range(0,len(contours)):
#     if contours[item].size > 500:
#         print("ping", contours[item].size, item )
#
# #cv2.drawContours(img,contours,73,(0,0,255),1)
# #cv2.drawContours(img,contours,97,(0,0,255),1)
# cv2.drawContours(img,contours,294,(0,0,255),1)



# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 7)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=7)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
ret, sure_fg = cv2.threshold(dist_transform,0.6*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
print(type(markers))
img[markers == -1] = [255,255,255]


gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#gray = cv2.equalizeHist(gray)
#ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
__,contours,__ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("gray",gray)

for item in range(0,len(contours)):
    if contours[item].size > 500:
        print("ping", contours[item].size, item )

cv2.drawContours(img, contours, 16966, (0, 0, 255), 1)
cv2.drawContours(img,contours,293,(0,0,255),1)
cv2.drawContours(img,contours,679,(0,0,255),1)

#################We finish image segmentation ##########################
# #ret,thresh = cv2.threshold(img,255,255,0)
# im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# #cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# M = cv2.moments(cnt)
#
#
# print( M )
#
#
# cx = int(M['m10']/M['m00'])
# cy = int(M['m01']/M['m00'])
#
# area = cv2.contourArea(cnt)
#
# perimeter = cv2.arcLength(cnt,True)
#
# epsilon = 0.1*cv2.arcLength(cnt,True)
# approx = cv2.approxPolyDP(cnt,epsilon,True)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()





#Sift
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(img,None)

img=cv2.drawKeypoints(img,kp,img)




img = cv2.imread('c_2.jpeg',cv2.IMREAD_COLOR)
surf = surf = cv2.xfeatures2d.SURF_create(400)
surf.setHessianThreshold(1500)
kp, des = surf.detectAndCompute(img,None)
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()





cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.meanShift()

