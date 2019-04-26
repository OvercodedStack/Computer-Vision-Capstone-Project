################################################################
#  Computer vision final project
#  Medical Lung Detection Program
#  Authors:
#        Esteban Segarra Martinez
#        Nicole Bruce
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
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels


filenames = glob.glob("TestingImages/Cancer/*.png")
filenames.sort()
images = [cv2.imread(img) for img in filenames]
DISPLAY_RESULTS = False
DO_HAVE_CANCER = True

class CompVisionFinal:
    threshold_dist =20.0
    def nothing(self,data):
        pass

    def __init__(self,img):
        # cv2.namedWindow('Image')
        # cv2.createTrackbar('Value 1','Image',0,255,self.nothing)
        # cv2.createTrackbar('Value 2','Image', 0, 255, self.nothing)
        orig = img
        if DISPLAY_RESULTS == True:
            cv2.imshow("Preview Image", img)
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
                #print(type(contours[item]))
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

        ################################################### Sift####################
        # sift = cv2.xfeatures2d.SIFT_create()
        # kp = sift.detect(out, None)
        ###########################################SURF#############################
        surf = surf = cv2.xfeatures2d.SURF_create(400)
        surf.setHessianThreshold(450)
        kp, des = surf.detectAndCompute(out,None)
        # img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
        # plt.imshow(img2),plt.show()

        ##########################################KEYPOINT MANIPULATION###############
        kp = self.clear_points(kp, list_contours)
        kp_cancer = self.check_cancer(kp)

        #########################TAKE DATA ###########################################
        self.kp_out = kp
        self.des_out = des
        self.samples_out = self.generate_samples(gray,kp)
        #########################DRAW DATA ############################################
        img = cv2.drawKeypoints(out,kp_cancer, None,(255,0,0))
        self.scaling_out = (img.shape[0]*img.shape[1]) / (orig.shape[0]*orig.shape[1])


        if DISPLAY_RESULTS == True:
            cv2.imshow('image',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        ############################################################################

    #Returns the keypoints of the image
    def get_kp(self):
        return self.kp_out

    def get_des(self):
        return self.des_out

    def get_samples(self):
        return self.samples_out

    def get_scaling_ratio(self):
        return self.scaling_out

    def take_snippet(selfs,point,image):
        size = 16, 16, 3
        sample = np.zeros(size, dtype=np.uint8)
        #print (point)
        for x in range(0, 16):
            for y in range (0,16):
                x_1 = int(x+point.pt[0]-8)
                y_1 = int(y+point.pt[1]-8)

                pixel_point = image[y_1,x_1]
                sample[x,y] = pixel_point
        return sample

    def generate_samples(self,image,kp):
        array_of_images = []
        for pt in kp:
            array_of_images.append(self.take_snippet(pt,image))
        return array_of_images

    def harris_detec(self,gray):
        # find Harris corners
        img = gray
        try:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)

        except:
            im2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

        dst = cv2.cornerHarris(gray, 1, 3 ,0.004)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        # Now draw them
        res = np.hstack((centroids, corners))
        res = np.int0(res)
        img[res[:, 1], res[:, 0]] = [0, 0, 255]
        img[res[:, 3], res[:, 2]] = [0, 255, 0]

        cv2.imshow('winname',img)
        cv2.waitKey(0)

        return corners

    #Calculates the elcucian distance between two points
    def calc_dist(self,point_1, point_2):
        return math.fabs(math.sqrt(math.pow(point_1.pt[0] - point_2.pt[0],2)  +   math.pow(point_1.pt[1] - point_2.pt[1],2) ))

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
        for contour in contour_list:
            for point in contour:
                for point_2 in range(0,len(kp)):
                    if self.in_circle(point,kp[point_2]):
                        temp_kp = np.delete(kp,point_2)
                kp = temp_kp
        return kp

    def use_FAST(self,gray):
        img2 = gray.copy
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(gray, None)
        img2 = cv2.drawKeypoints(gray,kp, None,(255,0,0))
        cv2.imshow('wow',img2)
        cv2.waitKey(0)


    #Check for cancer, if the amount of points is too high, ignore, if it's too low, also ignore
    def check_cancer(self, kp):
        list_of_nodes = []
        accumulator = 0
        acc_limit =5
        min_dist = 8
        max_dist = 30 #Pixels my guess
        for pt in kp:
            for pt_2 in kp:
                dist = self.calc_dist(pt,pt_2)
                if  dist <= max_dist and dist >= min_dist:
                    accumulator += 1
                if accumulator >= acc_limit:
                    accumulator = 0
                    break
            if accumulator != 0:
                 list_of_nodes.append(pt)
            accumulator = 0
        return list_of_nodes
########################################################################################################################




########################################################################################################################
class Learning:
    def __init__(self,images,filenames):
        training_labels = ["Nothing wrong", "Cancer", "Other Problems"]
        data_results = []
        counter = 0
        for img in images:
            img_data_info  = [None] * 3
            img_data_info[0] =filenames[counter]
            spotter = CompVisionFinal(img)
            des = spotter.get_des()

            print(len(spotter.get_kp()))
            print(spotter.get_scaling_ratio())
            if ( len(spotter.get_kp()) >= (900 * spotter.get_scaling_ratio() *(9000/45000) )):
                img_data_info[1] = 'True'
            else:
                img_data_info[1] = 'False'

            sampleImages = spotter.get_samples()
            self._Learning(des,5, sampleImages,training_labels)

            img_data_info[2] = self.accuracy_out
            data_results.append(img_data_info)
            print("================END RESULTS OF IMG %d ==============================================" % counter)

            counter += 1
        self.print_end_results(data_results)


    def print_end_results(self,img_data_results):
        print("===================Name of Image ------------- Is Dieseased? -- Accuracy %======================")
        tally = 0.0
        for item in img_data_results:
            if item[1] == 'True' and DO_HAVE_CANCER:
                tally += 1
            elif item[1] == 'False' and DO_HAVE_CANCER == False :
                tally += 1
            print ("Img: %s, R: %s , ACC: %f " % (item[0],item[1],item[2]))
        tally = tally/len(img_data_results)
        print("Percent total of correct results: %f" % tally)


    def _Learning(self,descriptor_list, n_clusters, n_images, train_labels, ret=None, std=None):
        # set up model
        kmeans_obj = KMeans(n_clusters=n_clusters)

        # format the data (descriptor list)
        vStack = np.array(descriptor_list[0])
        for remaining in descriptor_list[1:]:
            vStack = np.vstack((vStack, remaining))
        descriptor_vstack = vStack.copy()

        # perform clustering
        kmeans_ret = kmeans_obj.fit_predict(descriptor_vstack)

        # develop vocabulary
        mega_histogram = np.array([np.zeros(n_clusters) for i in range(0,len(n_images))])
        old_count = 0
        for i in range(0,len(n_images)):
            l = len(descriptor_list[i])
            for j in range(l):
                if ret is None:
                    try:
                        idx = kmeans_ret[old_count + j]
                    except:
                        pass
                else:
                    idx = ret[old_count + j]
                mega_histogram[i][idx] += 1
            old_count += 1
        print("Vocabulary Histogram Generated")

        if DISPLAY_RESULTS == True:
            # display trained vocabulary
            vocabulary = mega_histogram
            x_scaler = np.arange(n_clusters)
            y_scalar = np.array([abs(np.sum(vocabulary[:, h])) for h in range(n_clusters)])
            plt.bar(x_scaler, y_scalar)
            plt.xlabel("Visual Word Index")
            plt.ylabel("Frequency")
            plt.title("Complete Vocabulary Generated")
            plt.xticks(x_scaler + 0.4, x_scaler)
            plt.show()

        # standardize
        if std is None:
            scale = StandardScaler().fit(mega_histogram)
            mega_histogram = scale.transform(mega_histogram)
        else:
            print("An external STD has been supplied. Applying to histogram")
            mega_histogram = std.transform(mega_histogram)

        training_wheels = []
        for i in range(len(mega_histogram)):
            training_wheels.append(str(i))

        # train--USES SVC!!
        clf = SVC()
        clf.fit(mega_histogram, training_wheels)
        self._predictAndAccuracy(clf, mega_histogram, training_wheels)


    def _SVC(self):
        # split data into train/test datasets
        test_size = 0.20
        seed = 7
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size,
                                                                            random_state=seed)
        # make predictions on test dataset
        svm = SVC()
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
        self.accuracy_out = accuracy_score(Y_test, predictions)

    def _KMeansTemp(self):  # copied from the given website, used for reference for KMeansFinal
        # load data in

        # split data into train/test datasets
        n_samples = 1000
        n_features = 5
        n_clusters = 3
        X, Y = make_blobs(n_samples, n_features)

        # make predictions on test dataset
        ret = KMeans(n_clusters=n_clusters).fit_predict(X)
        __, ax = plt.subplots(2)
        ax[0].scatter(X[:, 0], X[:, 1])
        ax[0].set_title("Initial Scatter Distribution")
        ax[1].scatter(X[:, 0], X[:, 1], c=ret)
        ax[1].set_title("Colored Partition denoting Clusters")
        plt.show()

    def plotConfusionMatrix(self, y_true, y_pred, classes, cmap=plt.cm.colors):
        title = "Normalized confusion matrix"
        cm = confusion_matrix(y_true, y_pred)
        classes = classes[unique_labels(y_true, y_pred)]
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xtickslabels=classes, ytickslabels=classes,
               title=title, ylabel='True Label', xlabel='Predicted Label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
                fig.tight_layout()
                return ax


test_2 = Learning(images,filenames)


