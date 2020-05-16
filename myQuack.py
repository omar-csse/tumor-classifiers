
'''

2020 

Scaffolding code for the Machine Learning assignment. 
You should complete the provided functions and add more functions and classes as necessary.
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.
You are welcome to use the pandas library if you know it.


'''

import os 
import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (9976353, 'Omar', 'Alqarni'), (9497862, 'Mohammed', 'Alsaeed'), (10368493, 'Sohyb', 'Qasem') ]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    ##         "INSERT YOUR CODE HERE"    
    data = np.genfromtxt(dataset_path, delimiter=",", dtype=str)

    y = [data[:,1]]
    Y_list = [None] * len(y[0])   
    
    i = 0
    for Toumer in (y[0]):
        i+=1 
        if Toumer == "M":
            Y_list[i-1]=1
        else:
            Y_list[i-1]=0
        y = np.array(Y_list)

    x_list = data[:,2:]
    x = np.array(x_list, dtype=np.float64)

    return x, y

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DecisionTree_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    max_N_value = 350
    min_N_value = 1
    n_range = list(range(min_N_value,max_N_value))
    param_grid = dict(n_neighbors = n_range)
    # Fit the model and build it
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, param_grid)
    clf.fit(X_train,Y_train)
    #Show the best neighbours value
    print("\n\nbuild_NearrestNeighbours_classifier model: ")
    print ("Model best score: ",clf.best_score_)
    print ("Model best params: ", clf.best_params_)

    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    '''  
    Build a Neural Network classifier (with two dense hidden layers)  
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ## AND OTHER FUNCTIONS TO COMPLETE THE EXPERIMENTS
    ##         "INSERT YOUR CODE HERE"    


def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    
    '''
    in this function we will print the results which is confusion matrix
    
    @param 
    model: the specified model
    X_train: X_train[i,:] is the ith example
	Y_train: Y_train[i] is the class label of X_train[i,:]
    X_test: X_test of the datasets
	Y_train: Y_train of the datasets

    @return
    None
    '''

    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 2, 1)
    conf = plot_confusion_matrix(model, X_train, Y_train, normalize='true', ax=ax)
    conf.ax_.set_title('Training Set Performance')
    ax = fig.add_subplot(1, 2, 2)
    conf = plot_confusion_matrix(model, X_test, Y_test, normalize='true', ax=ax)
    conf.ax_.set_title('Validation Set Performance')
    pred = model.predict(X_test)
    pred1 = model.predict(X_train)
    print('Testing Accuracy: ' + str(sum(pred == Y_test)/len(Y_test)))
    print('Train Accuracy: ' + str(sum(pred1 == Y_train)/len(Y_train)))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here

    ##         "INSERT YOUR CODE HERE"    
    data_path = os.path.dirname(os.path.realpath(__file__)) + '/medical_records.data'
    x, y = prepare_dataset(data_path)
    #split the data randomly to 80% training set and 20% testting sets 
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.20)

    # call classifiers
    cknn = build_NearrestNeighbours_classifier(X_train, Y_train)
    evaluate_model(cknn, X_train, Y_train, X_test, Y_test)
    

