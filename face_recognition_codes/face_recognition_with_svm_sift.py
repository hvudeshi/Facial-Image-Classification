"""
    File name: face_recognition_with_svm_sift.py
    
    This code performs face recognition using SIFT features on ML face dataset constructed by students in the 'Machine Learning' course taught by Dr. Mehul Raval at School of Engineering and Applied Science, Ahmedabad University.

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

# dictionary of labels
rnoMapping = {}

def datagen(files):
    """
    Function: datagen 
    
    Input: List of filenames with their absolute paths
    Output: Training data and labels
    
    Description: This function computes SIFT features for each image in the Dataset/train folder, thresholds no. of SIFT feature vectors for an image, assigns label to each keypoint vector of the image and returns the final training data and labels matrices used for feeding the SVM in training phase.
    
    """

    Xtrain=[]
    ytrain=[]

    cnt = 0
    path = os.getcwd() + '/Dataset/train/'

    for filename in files:

        # read image
        img = cv2.imread(filename)

        # create SIFT object
        sift = cv2.xfeatures2d.SIFT_create()
        
        # compute SIFT keypoints & descriptors for images
        kp, des = sift.detectAndCompute(img, None)

        # procedure to extract actual label of image from filename
        onlyFnm = filename.replace(path, '')
        index = onlyFnm.index('_')
        # get actual label
        label = onlyFnm[:index]
                
        # construct dictionary for roll no. mapping
        if label not in rnoMapping.keys():
            rnoMapping[label] = cnt
            cnt += 1

        ndes = 0
        for d in des:
            if ndes <= 50: # threshold no. of SIFT keypoint vectors to include
                Xtrain.append(d.astype(float))
                ytrain.append(rnoMapping[label])
                ndes += 1
            else:
                break

    # return data and label
    return Xtrain, ytrain

def main():
    # list of training and test files
    files_train = [(os.getcwd() + '/Dataset/train/' + f) for f in os.listdir(os.path.join('./Dataset/','train'))]
    files_test = [(os.getcwd() + '/Dataset/test/' + f) for f in os.listdir(os.path.join('./Dataset/','test'))]

    # call 'datagen' function to get training and testing data & labels
    Xtrain, ytrain = datagen(files_train)

    # convert all matrices to numpy array for fast computation
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)

    # training phase: SVM , fit model to training data ------------------------------
    clf = svm.SVC(kernel = 'rbf', C = 10, gamma = 0.00001)
    clf.fit(Xtrain, ytrain)

    # create a SIFT feature object
    sift = cv2.xfeatures2d.SIFT_create()

    # testing phase & computing accuracy
    accuracy = 0
    path = os.getcwd() + '/Dataset/test/'

    for filename in files_test:
        img = cv2.imread(filename)
        
        # compute SIFT keypoints & descriptors for images
        kp, des = sift.detectAndCompute(img, None)

        # predict labels for keypoints of test image
        temp = clf.predict(des)
        pred = temp.astype(np.int64)

        # hash the labels assigned to keypoints & pick the label assigned to majority of the keypoints
        counts = np.bincount(pred)
        pred_label = np.argmax(counts)

        # procedure to extract actual label of image from filename
        onlyFnm = filename.replace(path, '')
        index = onlyFnm.index('_')
        label = onlyFnm[:index]
        
        # map actual label to the integer assigned to it
        actual_label = rnoMapping[label]

        # if ground truth is equal to the predicted label
        if actual_label == pred_label:
            accuracy +=1
        
        print(onlyFnm + " - Actual label: " + str(actual_label) + " - Pred label: " + str(pred_label))

    # calculate final accuracy; classes are balanced, hence accuracy is a good measure
    accuracy = (accuracy / len(files_test)) * 100
    print("\nAccuracy: %.2f" % accuracy + "%")

if __name__ == "__main__": 
    start_time = time.time()
    main()
    print("\n--- %s seconds ---" % (time.time() - start_time))