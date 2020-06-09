
'''

2020 

Scaffolding code for the Machine Learning assignment. 
You should complete the provided functions and add more functions and classes as necessary.
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.
You are welcome to use the pandas library if you know it.


'''

import os 
import itertools
import csv

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


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
    for Tumour in (y[0]):
        i+=1 
        if Tumour == "M":
            Y_list[i-1]=1
        else:
            Y_list[i-1]=0
        y = np.array(Y_list)

    x_list = data[:,2:]
    x = np.array(x_list, dtype=np.float64)

    return x, y, data

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
    dt = DecisionTreeClassifier()
    tree_depth = list(range(1, 32+1))
    param_grid = dict(max_depth=tree_depth)
    clf = GridSearchCV(dt, param_grid)
    clf.fit(X_training, y_training)

    print("\n\nbuild_DecisionTree_classifier model: ")
    print ("Model best score: ",clf.best_score_)
    print ("Model best params: ", clf.best_params_)

    scores = clf.cv_results_['mean_test_score']
    write_csv("DecisionTree_classifier", scores, tree_depth)

    return clf

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
    max_K_value = 100
    min_K_value = 1
    k_range = list(range(min_K_value, max_K_value+1))
    param_grid = dict(n_neighbors=k_range)
    # Fit the model and build it
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, param_grid)
    clf.fit(X_training, y_training)

    # Show the best neighbours value
    print("\n\nbuild_NearrestNeighbours_classifier model: ")
    print ("Model best score: ",clf.best_score_)
    print ("Model best params: ", clf.best_params_)

    scores = clf.cv_results_['mean_test_score']
    write_csv("NearrestNeighbours_classifier", scores, k_range)

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
    svmc = SVC()
    parameter_C = list(range(1, 10+1))
    param_grid = dict(kernel=['rbf'], C=parameter_C)
    clf = GridSearchCV(svmc, param_grid)
    clf.fit(X_training, y_training)

    print("\n\nbuild_SupportVectorMachine_classifier model: ")
    print ("Model best score: ",clf.best_score_)
    print ("Model best params: ", clf.best_params_)

    scores = clf.cv_results_['mean_test_score']
    write_csv("SupportVectorMachine_classifier", scores, parameter_C)

    return clf


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

    print("\n\n\n")
    # create model
    clf = tf.keras.Sequential()
    clf.add(tf.keras.layers.Flatten())
    clf.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    clf.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    clf.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    clf.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    clf.fit(X_training, y_training, shuffle=True, epochs=100, batch_size=20)

    return clf

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
    test_predictions = model.predict(X_test)
    train_predictions = model.predict(X_train)
    print('Testing Accuracy: ' + str(sum(test_predictions == Y_test)/len(Y_test)))
    print('Train Accuracy: ' + str(sum(train_predictions == Y_train)/len(Y_train)))

    show_prediction(model, data, X_test)


def evaluate_nn_model(model, X_train, Y_train, X_test, Y_test, data):

    val_loss, val_acc = model.evaluate(X_test, Y_test)
    print("\n\nbuild_NeuralNetwork_classifier model: ")
    print('Testing Accuracy: ' + str(val_acc))
    print('Testing loss: ' + str(val_loss))

    show_prediction(model, data, X_test)


def show_prediction(model, data, X_test):

    predictions = model.predict(X_test)
    classes = (predictions > 0.5).astype("int32")

    print("\n")

    for i, row in enumerate(data):
        for j, test_row in enumerate(X_test):
            current_row = np.array(row[2:], dtype=np.float64)
            if (current_row==test_row).all():
                if classes[j] == 1: Tumour = "M"
                else: Tumour = "B"

                msg = "id:{} - expeceted_tumour:{} || predicted_tumour:{}".format(row[0], row[1], Tumour)
                if Tumour != row[1]: msg = msg + " (**WRONG**)"
                print(msg)
    

def write_csv(fname, xs, ys):
    with open(fname+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x-axis", "y-axis"])

        for i in range(0, len(xs)):
            writer.writerow([xs[i], ys[i]])


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":

    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here

    ##         "INSERT YOUR CODE HERE"
    data_path = os.path.dirname(os.path.realpath(__file__)) + '/medical_records.data'
    x, y, data = prepare_dataset(data_path)
    # split the data randomly to 80% training set and 20% testting sets 
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=42, test_size=0.20)

    # call classifiers
    dt_c = build_DecisionTree_classifier(X_train, Y_train)
    evaluate_model(dt_c, X_train, Y_train, X_test, Y_test)

    knn_c = build_NearrestNeighbours_classifier(X_train, Y_train)
    evaluate_model(knn_c, X_train, Y_train, X_test, Y_test)

    svm_c = build_SupportVectorMachine_classifier(X_train, Y_train)
    evaluate_model(svm_c, X_train, Y_train, X_test, Y_test)

    nn_c = build_NeuralNetwork_classifier(X_train, Y_train)
    evaluate_nn_model(nn_c, X_train, Y_train, X_test, Y_test, data)
    

