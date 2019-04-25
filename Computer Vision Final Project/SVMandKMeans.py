# Load ML libraries
import sys
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn import KMeans
from sklearn.svm import SVC
	
def _main():
	# load dataset
	#dunno what put here, depends on how being fed in
	#will have X and Y (x= features being trained, y=feature we wish to predict
	
	# split data into train/test datasets
	test_size = 0.20
	seed = 7
	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
	
	# make predictions on test dataset
	svm = SVMClassifier()
	_train(svm, X_train, Y_train, X_test, Y_test)
	
	#If using KMeans
	
		
#####################################################
# training and prediction functions
def _train(alg, X_train, Y_train, X_test, Y_test):
	alg.fit(X_train, Y_train)
	_predictAndAccuracy(alg,X_test,Y_test)

def _predictAndAccuracy(alg, X_test, Y_test):
	predictions = alg.predict(X_test)
	print(confusion_matrix(Y_test, predictions))
	print("Accuracy: %0.2f" % (accuracy_score(Y_test, predictions)))

_main()
