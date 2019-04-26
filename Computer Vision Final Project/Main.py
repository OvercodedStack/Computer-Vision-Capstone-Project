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
import glob
import sys
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn import KMeans
from sklearn.svm import SVC

filenames = glob.glob("TestingImages/Cancer/*.png")
filenames.sort()
images = [cv2.imread(img) for img in filenames]
class CompVisionFinal:
    threshold_dist =20.0
    def nothing(self,data):
        pass

    def __init__(self,img):
        # cv2.namedWindow('Image')
        # cv2.createTrackbar('Value 1','Image',0,255,self.nothing)
        # cv2.createTrackbar('Value 2','Image', 0, 255, self.nothing)
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

        ######################CROP OUT THE LUNG IMAGES##############################
        out = np.zeros((height, width, 3), np.uint8)
        list_contours = []
        counter = 0
        for item in range(0,len(contours)):
            if contours[item].size > graded_size:
                list_contours.append(contours[item])
                print(type(contours[item]))
                list_contours[counter] = contours[item]
                cv2.drawContours(img, contours, item, (0, 0, 255), 1)
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

        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(img)
        gray = cv2.drawKeypoints(out, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print (keypoints)
        cv2.imshow("gray", gray)
        cv2.waitKey(0)
        ################################################### Sift####################
        # sift = cv2.xfeatures2d.SIFT_create()
        # kp = sift.detect(out, None)
        ###########################################SURF#############################
        surf = surf = cv2.xfeatures2d.SURF_create(400)
        surf.setHessianThreshold(500)
        kp, des = surf.detectAndCompute(out,None)
        img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
        plt.imshow(img2),plt.show()
        ##########################################KEYPOINT MANIPULATION###############

        kp = self.clear_points(kp, list_contours)
        self.kp_out = kp
        img = cv2.drawKeypoints(out, kp, img)
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ############################################################################

    #Returns the keypoints of the image
    def get_kp(self):
        return self.kp_out

    #Calculates the elcucian distance between two points
    def calc_dist(self,point_1, point_2):
        return math.fabs(math.sqrt(math.pow(point_1[0][0][0] - point_2.pt[0],2)  +    math.pow(point_1[0][0][1] - point_2.pt[1],2) ))

    #Checks if a point is inside a range of a circle distance
    def in_circle(self,point_2, point_1):
        radius = self.threshold_dist
        center_x = point_1.pt[0]
        center_y = point_1.pt[1]
        x_2 = point_2[0][0]
        y_2 = point_2[0][1]
        dist = math.sqrt((center_x - x_2) ** 2 + (center_y - y_2) ** 2)
        return dist <= radius

    #Delete keypoints near the contour line.
    def clear_points(self,kp,contour_list):
        temp_kp = kp
        print(len(contour_list))
        for contour in contour_list:
            print(contour)
            for point in contour:
                for point_2 in range(0,len(kp)):
                    if self.in_circle(point,kp[point_2]):
                        temp_kp = np.delete(kp,point_2)
                kp = temp_kp
        return kp

#Runs for all loaded images
for img in images:
    test_1 = CompVisionFinal(img)


class Learning:
    def __init__(self):
        # load dataset
        # dunno what put here, depends on how being fed in
        # will have X and Y (x= features being trained, y=feature we wish to predict

        # split data into train/test datasets
        test_size = 0.20
        seed = 7
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size,
                                                                            random_state=seed)

        # make predictions on test dataset
        svm = SVMClassifier()
        self._train(svm, X_train, Y_train, X_test, Y_test)

    # If using KMeans


    #####################################################
    # training and prediction functions
    def _train(self,alg, X_train, Y_train, X_test, Y_test):
        alg.fit(X_train, Y_train)
        self._predictAndAccuracy(alg, X_test, Y_test)

    def _predictAndAccuracy(self,alg, X_test, Y_test):
        predictions = alg.predict(X_test)
        print(confusion_matrix(Y_test, predictions))
        print("Accuracy: %0.2f" % (accuracy_score(Y_test, predictions)))

    # test_1 = CompVisionFinal(img)


#img = cv2.imread('c_2.jpeg',cv2.IMREAD_COLOR)
#
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#


    # cv2.drawContours(gray, contours, -1, (0, 255, 0), 3)
    # cv2.imshow('a',gray)
    # cv2.waitKey(0)
    #
    # for point in range(0, len(kp)):
    #     for point_2 in range(0,len(list_contours)):
    #         if point > list_contours[point_2]:


    # cv2.imshow("gray",gray)
    # for contour in contours:
    #     M = cv2.moments(contour)
    #     if M["m00"] != 0:
    #         cX = int(M["m10"] / M["m00"])
    #         cY = int(M["m01"] / M["m00"])
    #     else:
    #         cX, cY = 0, 0
    #     cv2.circle(gray, (cX, cY), 5, (255, 255, 255), -1)
    #     # cv2.putText(gray, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.imshow('img', gray)
    # cv2.waitKey(0)





# gray = cv2.Canny(gray, 128,255)
# while(1):
#     val_1 = cv2.getTrackbarPos('Value 1','Image')
#     val_2 = cv2.getTrackbarPos('Value 2', 'Image')
#     test = cv2.Canny(gray,val_1,val_2)
#     cv2.imshow('Image',test)
#     cv2.waitKey(1)
