
"""
FaceRecognition - Main SVM

"""
#__author__ = "Nawel Medjkoune, Fella Belkham"


from __future__ import division
from DataSet import LabelData
from extract_descriptors import extract_sift
from extract_descriptors import extract_surf
from extract_descriptors import extract_hog
from SVM import SVMClassifyer
import numpy as np
import random
import sklearn
from sklearn import svm
import cv2 
from image_processing import *
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn.metrics import classification_report,accuracy_score
from matplotlib import pyplot as plt

"""
Testing the SIFT Descriptors

"""
print("Getting the data ...")
data  = LabelData("orl")
descriptors = []
labels = []
descript_hog=[]

print("Applying Hog ...")
for img in data:
    im=cv2.imread(img[0])
    #print im
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im_gray = cv2.resize(im_gray,(128,128))
    des, hog_img = extract_hog(im_gray)
    cnt=0;
    descriptors.append(des)
    #print(des)
    #descriptors.append(des)
    # for d in des :
    #     #descriptors = np.vstack([descriptors,d])
    #     # d = np.transpose(d)
    #     descriptors.append(d)
    #     #descript_hog=np.vstack([descript_hog,d])
    #                 #print(img[1])
    #     #print("\n")
    #     #cnt=cnt+1
    labels.append(img[1])

#descriptors = np.resize(descriptors,(135,512))        
# descriptors = np.array(descriptors)
print("Training the classifyer ...")
#descriptors=np.transpose(descriptors)

labels = np.array(labels).reshape(len(labels),1)
#print(descriptors)

hog_features = np.array(descriptors)
hog_features.astype('float64')
clf = svm.SVC(kernel="linear", C=100.0, gamma ="auto") #initialising the svm #classifier
print(hog_features.shape)
print(labels.shape)
data_frame = np.hstack((hog_features,labels))
#print(data_frame.shape)
np.random.shuffle(data_frame)

percentage = 70
partition = int(len(hog_features)*percentage/100)

x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()

# print(y_train.shape)
# print(y_train)
x_train.astype('float64')
y_train.astype('float64')

clf.fit(x_train,y_train)

score_train = clf.score(x_train,y_train)
print('\n')
print("Final HOG Accuracy:")
print("Score of Training:"+str(score_train))

score = clf.score(x_test,y_test)
print("Score:"+str(score))

# for img in test:
    
#     real_label = img[1]
#     im=cv2.imread(img[0])
#     im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#     des, hog_img = extract_hog(im_gray)
#     temp = svm.predict(des)
#     #pred = svm.predict(des)
#     pred = temp.astype(np.int64)
#     # print(pred)
#     # print("\n")
#     counts = np.bincount(pred)
#     # print(counts)
#     # print("\n")
#     pred_label = np.argmax(counts)

#     if real_label == str(pred_label) :
#         accuracy +=1
#     print img[0] + " - Real labe: " + str(real_label) + " - Pred label:" + str(pred_label)
# accuracy = accuracy/total_imgs*100
# print "Accuracy obtained using HOG is: %.2f" % accuracy + "%"

"""
Testing variation on the SIFT Descriptors

"""
# listFunctions =["gaussian_blur","median_blur","brightness","darker","erosion","dilation","gaussian_noise","salt_pepper"]
# accuracy = 0
# total_imgs = len(test)

# for fnc in listFunctions: 
#     print "Testing Surf with " + fnc
#     for img in test:
#             real_label = img[1]
#             im=cv2.imread(img[0])
#             im = globals()[fnc](im)
#             im=im.astype('uint8') 
#             kp, des = extract_sift(im)
#             pred = svm.predict(des)
#             counts = np.bincount(pred)
#             pred_label = np.argmax(counts)
#             if real_label == pred_label :
#                 accuracy +=1        
#         #print img[0] + " - Real labe: " + str(real_label) + " - Pred label:" + str(pred_label)
#     accuracy = accuracy/total_imgs*100
#     print "Accuracy obtained after applying "+ fnc + " is: %.2f" % accuracy + "%"


# """
# Testing the SURF Descriptors

# """
# print 'Getting the data ...'
# train, test  = LabelData("orl")
# descriptors = []
# labels = []

# print 'Applying Sift ...'
# for img in train:
#     im=cv2.imread(img[0])
#     kp, des = extract_surf(im)
#     for d in des :
#         descriptors.append(d)
#         labels.append(img[1])
        
# c = list(zip(descriptors, labels))
# random.shuffle(c)
# descriptors, labels = zip(*c)

# print "Training the classifyer ..."
# svm = SVMClassifyer("rbf", 10000, 0.1, descriptors, labels)

# accuracy = 0
# total_imgs = len(test)

# for img in test:
    
#     real_label = img[1]
#     im=cv2.imread(img[0])
#     kp, des = extract_surf(im)
#     pred = svm.predict(des)
#     counts = np.bincount(pred)
#     pred_label = np.argmax(counts)
#     if real_label == pred_label :
#         accuracy +=1
#     #print img[0] + " - Real labe: " + str(real_label) + " - Pred label:" + str(pred_label)
# accuracy = accuracy/total_imgs*100
# print "Accuracy obtained is: %.2f" % accuracy + "%"


# """
# Testing variation on the SURF Descriptors

# """
# listFunctions =["gaussian_blur","median_blur","brightness","darker","erosion","dilation","gaussian_noise","salt_pepper"]
# accuracy = 0
# total_imgs = len(test)

# for fnc in listFunctions: 
#     print "Testing Surf with " + fnc
#     for img in test:
#             real_label = img[1]
#             im=cv2.imread(img[0])
#             im = globals()[fnc](im)
#             im=im.astype('uint8') 
#             kp, des = extract_surf(im)
#             pred = svm.predict(des)
#             counts = np.bincount(pred)
#             pred_label = np.argmax(counts)
#             if real_label == pred_label :
#                 accuracy +=1        
#         #print img[0] + " - Real labe: " + str(real_label) + " - Pred label:" + str(pred_label)
#     accuracy = accuracy/total_imgs*100
#     print "Accuracy obtained after applying "+ fnc + " is: %.2f" % accuracy + "%"

