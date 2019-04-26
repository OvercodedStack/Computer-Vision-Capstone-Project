# Load ML libraries
import sys
import numpy as np
import matplotlib as plt

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn import KMeans
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
	
def _SVC():
	# load dataset
	#dunno what put here, depends on how being fed in
	#will have X and Y (x= features being trained, y=feature we wish to predict
	
	# split data into train/test datasets
	test_size = 0.20
	seed = 7
	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
	
	# make predictions on test dataset
	svm = SVC()
	_train(svm, X_train, Y_train, X_test, Y_test)
	
def _KMeansTemp(): #copied from the given website, used for reference for KMeansFinal
	#load data in
	
	#split data into train/test datasets
	n_samples = 1000
	n_features = 5
	n_clusters = 3
	X,Y = make_blobs(n_samples,n_features)
	
	#make predictions on test dataset
	ret = KMeans(n_clusters = n_clusters).fit_predict(X)
	print ret
	
	__,ax = plt.subplots(2)
	ax[0].scatter(X[:,0],X[:,1])
	ax[0].set_title("Initial Scatter Distribution")
	ax[1].scatter(X[:,0], X[:,1], c=ret)
	ax[1].set_title("Colored Partition denoting Clusters")
	plt.show()

def _Learning(descriptor_list, n_clusters, n_images, train_labels, ret=None, std=None):
	#set up model
	kmeans_obj = KMeans(n_clusters=n_clusters)
	
	#format the data (descriptor list)
	vStack = np.array(descriptor_list[0])
	for remaining in descriptor_list[1:]:
		vStack = np.vstack((vStack, remaining))
	descriptor_vstack = vStack.copy()
	
	#perform clustering
	kmeans_ret = kmeans_obj.fit_predict(descriptor_vstack)
	
	#develop vocabulary
	mega_histogram = np.array([np.zeros(n_clusters) for i in range(n_images)])
	old_count = 0
	for i in range(n_images):
		l = len(descriptor_list[i])
		for j in range(l):
			if ret is None:
				idx = kmeans_ret[old_count+j]
			else:
				idx = ret[old_count+j]
			mega_histogram[i][idx]+=1
		old_count+=1
	print "Vocabulary Histogram Generated"
	
	#standardize
	if std is None:
		scale = StandardScaler().fit(mega_histogram)
		mega_histogram = scale.transform(mega_histogram)
	else:
		print "An external STD has been supplied. Applying to histogram"
		mega_histogram = std.transform(mega_histogram)
	
	#train--USES SVC!!
	print "Training the SVM Model"
	print "Training labels: ",train_labels
	clf = SVC()
	self.clf.fit(mega_histogram,train_labels)
	print "Training has completed"
	
		
#####################################################
# training and prediction functions--FOR SVC
def _train(alg, X_train, Y_train, X_test, Y_test):
	alg.fit(X_train, Y_train)
	_predictAndAccuracy(alg,X_test,Y_test)

def _predictAndAccuracy(alg, X_test, Y_test):
	predictions = alg.predict(X_test)
	print(confusion_matrix(Y_test, predictions))
	print("Accuracy: %0.2f" % (accuracy_score(Y_test, predictions)))

_main()
