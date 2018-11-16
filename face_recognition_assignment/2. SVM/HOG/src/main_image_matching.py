"""
FaceRecognition - Main image matching 

This code aims to reproduce the first part of our experiments: Features matching across different facile expression and configuration
Dataset used: Yale, subject 9
"""
__author__ = "Nawel Medjkoune, Fella Belkham"


import numpy as np
import cv2
from matplotlib import pyplot as plt
import drawMatches as dm
import extract_descriptors as ed
import pickle

files=['happy', 'sad', 'sleepy', 'wink','surprised','glasses', 'centerlight', 'leftlight','rightlight']

######################################################## ################################### ########################## #############################
#SURF features


saved_surf=[]
for f in files:
    path1='../Dataset/subject9/subject09.normal.png'
    path2='../Dataset/subject9/subject09.'+f+'.png'

    savename='surf_'+f+'.png'

    # find the keypoints and descriptors with SURF
    kp1, des1 = ed.extract_surf(path1, hessian=500, octave=3, octaveLayers=2, ext=False)
    kp2, des2 = ed.extract_surf(path2, hessian=500, octave=3, octaveLayers=2, ext=False)
    
    #find good matches
    good=ed.matcher(kp1, kp2, des1, des2)

    #draw matches
    img3 = dm.drawMatches(cv2.imread(path1,0),kp1,cv2.imread(path2,0),kp2,good)
    
    #write image
    cv2.imwrite(savename,img3)
    print float(len(good))/float(min(len(des1),len(des2)))

    #compute mispatching percentge
    tmp=[img3, float(len(good))/float(min(len(des1),len(des2))), savename]
    saved_surf.append(tmp)

pickle.dump(saved_surf,open("saved_surf", "wb"))


######################################################## ################################### ########################## #############################
#SIFT features

saved_sift=[]
for f in files:
    path1='../Dataset/subject9/subject09.normal.png'
    path2='../Dataset/subject9/subject09.'+f+'.png'

    savename='sift_'+f+'.png'

    # find the keypoints and descriptors with SIFT
    kp1, des1 = ed.extract_sift(path1,  octave=3, contrast=0.03, edge=10, sigma=1.6)
    kp2, des2 = ed.extract_sift(path2,  octave=3, contrast=0.03, edge=10, sigma=1.6)

    good=ed.matcher(kp1, kp2, des1, des2)

    img3 = dm.drawMatches(cv2.imread(path1,0),kp1,cv2.imread(path2,0),kp2,good)

    cv2.imwrite(savename,img3)
    print float(len(good))/float(len(des1))

    tmp=[img3, float(len(good))/float(min(len(des1),len(des2))), savename]
    saved_sift.append(tmp)

pickle.dump(saved_sift,open("saved_sift", "wb")) 
