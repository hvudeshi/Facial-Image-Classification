"""
    File name: face_recognition_with_svm_hog.py
    
    This code performs face recognition using HOG features on ML face dataset constructed by students in the 'Machine Learning' course taught by Dr. Mehul Raval at School of Engineering and Applied Science, Ahmedabad University.

    Author: Chintan Gandhi
"""

import numpy as np
import pandas as pd
import cv2
import os
import sys
import math
import time
from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# dictionary of labels
rnoMapping = {}

def datagen(files, mode):
    """
    Function: datagen 
    
    Input: 
        List of filenames with their absolute paths
        mode - 1 denotes train data ; 2 denotes test data
    
    Output: Train/Test data and labels depending on mode value
    
    Description: This function computes HOG features for each image in the Dataset/train or Dataset/test folder, assigns label to the descriptor vector of the image and returns the final train/test data and labels matrices used for feeding the SVM in training phase or predicting the label of test data.
    
    """

    X = []
    y = []

    cnt = 0

    for filename in files:

        # read image
        img = cv2.imread(filename)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(128,128))

        # compute HOG features
        des, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4), block_norm= 'L2',visualize=True)

        if mode == 1:
            path = os.getcwd() + '/Dataset/train/'
        elif mode == 2:
            path = os.getcwd() + '/Dataset/test/'
        
        # procedure to extract actual label of image from filename
        onlyFnm = filename.replace(path, '')
        index = onlyFnm.index('_')
        label = onlyFnm[:index]
                
        if mode == 1:
            # construct dictionary for roll no. mapping
            if label not in rnoMapping.keys():
                rnoMapping[label] = cnt
                cnt += 1

        # append descriptor and label to train/test data, labels
        X.append(des)
        y.append(rnoMapping[label])

    # return data and label
    return X, y

def main():
    # list of training and test files
    files_train = [(os.getcwd() + '/Dataset/train/' + f) for f in os.listdir(os.path.join('./Dataset/','train'))]
    files_test = [(os.getcwd() + '/Dataset/test/' + f) for f in os.listdir(os.path.join('./Dataset/','test'))]

    # call 'datagen' function to get training and testing data & labels
    Xtrain, ytrain = datagen(files_train, 1)
    Xtest, ytest = datagen(files_test, 2)

    # convert all matrices to numpy array for fast computation
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    Xtest = np.array(Xtest)
    ytest = np.array(ytest)

    # training phase: SVM , fit model to training data ------------------------------
    clf = svm.SVC(kernel = 'linear')
    clf.fit(Xtrain, ytrain)
    # predict labels for test data
    ypred = clf.predict(Xtest)
    
    # compute accuracy
    accuracy = accuracy_score(ytest, ypred) * 100
    print("\nAccuracy: %.2f" % accuracy + "%")

if __name__ == "__main__": 
    start_time = time.time()
    main()
    print('Execution time: %.2f' % (time.time() - start_time) + ' seconds\n')