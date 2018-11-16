
"""
FaceRecognition - Main SVM

"""

from __future__ import division
from DataSet import LabelData
from extract_descriptors import extract_sift
from extract_descriptors import extract_surf
from SVM import SVMClassifyer
import numpy as np
import pandas as pd
import random
import cv2 
from image_processing import *

"""
Testing the SIFT Descriptors

"""
print("Getting the data ...")
train, test  = LabelData("orl")
descriptors = []
labels = []

def normFunc(row):
    row['norm'] = np.sqrt(np.square(row).sum())
    return row

print("Applying Sift ...")
for img in train:
    im=cv2.imread(img[0])
    kp, des = extract_sift(im)
    cnt=0;
    
    des_df = pd.DataFrame(np.array(des))
    des_df = des_df.apply(normFunc, axis=1)
    des_df.sort_values(by=['norm'],ascending=False,inplace=True)
    des_df.drop(['norm'], axis=1, inplace=True)
    des = des_df.values
    
    for d in des :
        if cnt<=50:
            descriptors.append(d)
            print(img[1])
            print("\n")
            cnt=cnt+1
            labels.append(img[1])
        
c = list(zip(descriptors, labels))
random.shuffle(c)
descriptors, labels = zip(*c)

print("Training the classifyer ...")

svm = SVMClassifyer("rbf", 10, 0.00001, descriptors, labels)

accuracy = 0
total_imgs = len(test)
#
for img in test:
    
    real_label = img[1]
    im=cv2.imread(img[0])
    kp, des = extract_sift(im)
    temp = svm.predict(des)
    #pred = svm.predict(des)
    pred = temp.astype(np.int64)
    # print(pred)
    # print("\n")
    counts = np.bincount(pred)
    # print(counts)
    # print("\n")
    pred_label = np.argmax(counts)

    if real_label == str(pred_label) :
        accuracy +=1
    print(img[0] + " - Real label: " + str(real_label) + " - Pred label:" + str(pred_label))
accuracy = accuracy/total_imgs*100
print("Accuracy obtained using SIFT is: %.2f" % accuracy + "%")

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

