import numpy as np
import csv
from math import floor
from numpy.linalg import multi_dot, inv, det
from random import randrange
import sys

def loadData(path_train, path_test):

	X_tra = np.genfromtxt(path_train, delimiter=',')[1:]
	X_tst = np.genfromtxt(path_test, delimiter=',')[1:]

	return X_tra, X_tst



def writeOutput(predictions, path_to_write):

	n = predictions.size
	ids = np.arange(1,n+1).reshape((n,1))
	predictions = predictions.reshape((n,1))

	concat = np.concatenate((ids, predictions), axis=1).astype(np.int32)

	f = open(path_to_write, "w")
	f.write("ID,Predicted\n")
	for line in concat:
		f.write(str(line[0])+","+str(line[1])+"\n")
	f.close()



def preprocessing(X_tra, treshold):  #return X_tra_new and real_eigen_val for dimention reduction in X_tst

	X_label = X_tra[:,-1]
	X_norm = X_tra[:,:-1] - np.mean(X_tra[:,:-1], axis=0)
	covar_mtx = np.dot(X_norm.T, X_norm)/X_norm.shape[1]

	#U, S, V_T = np.linalg.svd(covar_mtx)
	eigen_val, eigen_vec = np.linalg.eig(covar_mtx)

	real_eigen_val = eigen_val.real.astype(float)
	real_eigen_val = real_eigen_val.reshape((1, real_eigen_val.size))
	real_eigen_vec = eigen_vec.real.astype(float)

	summ = np.sum(np.abs(real_eigen_val))
	total, r_dim = 0, 0  #total for accumulation and r_dim will hold the reduced dimention
	for i in np.squeeze(real_eigen_val[:,real_eigen_val.argsort()[::-1]])[::-1]:
		total += 100*i/summ
		r_dim += 1
		if total >= treshold:
			break
	print("reduced to the: ", r_dim, " feature")

	cat_X_tra = np.concatenate((real_eigen_val, X_tra[:,:-1]), axis=0)		#concatenate eigen rows in order to sort by this row
	reduced_X_tra = cat_X_tra[:,cat_X_tra[0,:].argsort()[::-1]][1:,:r_dim] #eigen value row and columns after kth column are truncated

	cat_real_eigen_vec = np.concatenate((real_eigen_val, real_eigen_vec), axis=0)	#concatenate eigen rows in order to sort by this row
	col_reduced_eigen_vec = cat_real_eigen_vec[:,cat_real_eigen_vec[0,:].argsort()[::-1]][1:,:r_dim]	#sort respect to eigen values and truncate columns after kth column
	cat_reduced_eigen_vec = np.concatenate((real_eigen_val.T, col_reduced_eigen_vec), axis=1)			#concatenate eigen column in order to sort by this column
	reduced_eigen_vec = cat_reduced_eigen_vec[cat_reduced_eigen_vec[0,:].argsort()[::-1], :][:r_dim, 1:]	#sort and truncate rows after kth row

	new_X_tra = np.dot(reduced_X_tra, reduced_eigen_vec)
	new_X_tra = np.concatenate((new_X_tra, X_label.reshape((X_label.size, 1))), axis=1)

	return new_X_tra, real_eigen_val, r_dim



def Cost(X, Theta):		#in order to tune learning rate

	n = X.shape[0]
	sigm = sigmoid(np.dot(X[:,:-1],Theta))

	cost = -(np.dot(X[:,-1], np.log(sigm)) + np.dot(1-X[:,-1], np.log(1-sigm)))/n

	print(cost)

##############################################	LOGISTIC REGRESSION STARTS	##########################################

sigmoid = lambda vector : 1/(1+np.exp(-vector))	#sigmoid function which can take numpy array as parameter

def logistic_regression(X, learning_rate, max_iter):

	n = X.shape[0] #number of samples
	bias_unit = np.ones(n).reshape((n,1))	#add bias unit (column of 1's)

	cat_X = np.concatenate((bias_unit, X), axis=1) #cat_X contains bias unit on its first row

	d = cat_X.shape[1] - 1 	#label is not included
	Theta = np.zeros(d)	#initialize Theta

	const = learning_rate/n

	for i in range(max_iter):
		Theta -= const * np.dot(cat_X[:,:-1].T, (sigmoid(np.dot(cat_X[:,:-1],Theta)) - cat_X[:,-1]))	#Gradient Descent

	return Theta

###############################################  LOGISTIC REGRESSION ENDS	###########################################

##############################################	K-Nearest Neighbor	Starts	###########################################

def KNN(train_set, test_set, K):

	predict = np.zeros(test_set.shape[0])
	treshold = floor(K/2)	#to measure majority

	for i, point in enumerate(test_set):

		dist_vectors = train_set[:,:-1] - point	# sample(without label)	- point
		distances = np.sqrt(np.sum(np.multiply(dist_vectors, dist_vectors), axis=1))	#square root of element-wise multiplication
		label_N_nearest = train_set[distances.argsort()[:K]][:,-1]		#select label of N points which have minimum distance

		if np.count_nonzero(label_N_nearest) > treshold:	# number of 1's greater than 0's
			predict[i] = 1

	return predict

##############################################	K-Nearest Neighbor	ENDS	###########################################

def regression_predict(X_test, model):

	n = X_test.shape[0] #number of samples
	bias_unit = np.ones(n).reshape((n,1))
	cat_X = np.concatenate((bias_unit, X_test), axis=1) #cat_X contains bias unit on its first row

	pred = np.array([1.0 if x else 0.0 for x in sigmoid(np.dot(cat_X, model)) >= 0.5])

	return pred

########################################### K-fold Cross Validation Starts ########################################


def cross_valid(data_set, fold_number, alpha , max_iter):

	N = data_set.shape[0]	#number of samples
	fold_size = floor(N/fold_number)	#size of each fold
	accuracy = []

	for i in range(fold_number):
		test_fold = data_set[i*fold_size : (i+1)*fold_size, :]
		x_test = test_fold[:,:-1]
		label_test = test_fold[:,-1]

		rest = np.concatenate((data_set[:i*fold_size, :], data_set[(i+1)*fold_size:, :]), axis=0)

		fold_model = logistic_regression(rest, alpha, max_iter)
		fold_pred = regression_predict(x_test, fold_model)

		num_correct = np.count_nonzero(fold_pred == label_test)
		accuracy.append(num_correct/fold_size)

	return 100*sum(accuracy)/fold_number	#return avarage accuracy


########################################### K-fold Cross Validation ENDS ########################################

def bagging(dataset, ratio=1.0):		#for bagging
	bag = []
	n_sample = round(len(dataset) * ratio)	#create a ratio*size of a bag and push uniformly random variables
	while len(bag) < n_sample:
		index = randrange(len(dataset))
		bag.append(dataset[index])

	return np.array(bag)


# 5 bag Bagging method		--	ENSEMBLE LEARNING  -- AT LEAST A TRY :D
def trainModel(X_train, X_tst, alpha=8.0, bag_ratio=1.0):		#X_tst given for KNN

	fold_number = 5
	#alpha = 8	#learning rate
	iter_1, iter_2 = 12000, 10000

	#bag_1 = bagging(X_train, bag_ratio)	#logistic regression with iter: 10000
	model_1 = logistic_regression(X_train, alpha, iter_1)
	pred_1 = regression_predict(X_tst, model_1)

	acc1 = cross_valid(X_train, fold_number, alpha, iter_1)
	print("accuracy of logistic regression for iter: ", iter_1, "is :", acc1)

	#bag_2 = bagging(X_train, bag_ratio)	#logistic regression with iter: 100000
	model_2 = logistic_regression(X_train, alpha, 12000)
	pred_2 = regression_predict(X_tst, model_2)

	acc2 = cross_valid(X_train, fold_number, alpha, iter_2)
	print("accuracy of logistic regression for iter: ", iter_2, "is :", acc2)

	#bag_3 = bagging(X_train, bag_ratio)	#KNN with 1 neighbor check
	pred_3 = KNN(X_train, X_tst, 3)

	#bag_4 = bagging(X_train, bag_ratio)	#KNN with 3 neighbor check
	pred_4 = KNN(X_train, X_tst, 5)

	#bag_5 = bagging(X_train, bag_ratio)	#KNN with 5 neighbor check
	pred_5 = KNN(X_train, X_tst, 1)

	return np.array([pred_1,pred_2, pred_3, pred_4, pred_5])



def predict(pred_list):	#Majority voting

	predict = np.zeros(pred_list.T.shape[0])	#initialize
	votes = np.count_nonzero(pred_list.T, axis=1)

	for i, vote in enumerate(votes):	#if 1's votes are majority predict 1
		if vote >= 2:
			predict[i] = 1

	return predict


#fold_number = 5
#acc = cross_valid_regression(X_new, fold_number, 8, 10000)
#print(acc)


if __name__ == "__main__":

	arguments = np.array(sys.argv[1:])

	treshold = float(arguments[0]) # reduced dimension will be contains more than %treshold of the information
	learning_rate = float(arguments[1])
	ratio = float(arguments[2])

	path = "sampleSubmission.csv"
	X_tra, X_tst = loadData('train.csv', "test.csv")
	X_new, eigen, r_dim = preprocessing(X_tra, treshold)	#X_new is reduced X_train

	cat_X_tst = np.concatenate((eigen, X_tst), axis=0)
	reduced_X_tst = cat_X_tst[:, cat_X_tst[0,:].argsort()[::-1]][1:,:r_dim]

	model = trainModel(X_new, reduced_X_tst, learning_rate, ratio)
	prediction = predict(model)

	#print(model)	#observe for voting

	writeOutput(prediction, path)
