"""
FaceRecognition - Image processing

"""
__author__ = "Nawel Medjkoune, Fella Belkham"


import cv2 
import numpy as np

def gaussian_blur(img,kernel=(5,5)):
    """
    applied gaussian blur to image 
    """
    return cv2.GaussianBlur(img,kernel,0) 

def median_blur(img,kernel=5):
    """
    applied median blur to image 
    """
    return cv2.medianBlur(img,kernel) #median blur

def brightness(img,phi = 2, theta = 2, maxIntensity = 255.0):
    """
    expands brightness of image
    """
    return (maxIntensity/phi)*(img/(maxIntensity/theta))**0.5

def darker(img, phi = 2, theta = 2, maxIntensity = 255.0):
    """
    reduce brightness of image
    """
    return (maxIntensity/phi)*(img/(maxIntensity/theta))**2

def erosion(img):
    kernel = np.ones((3,3),np.uint8)
    return cv2.erode(img,kernel,iterations = 1)

def dilation(img):
    kernel = np.ones((3,3),np.uint8)
    return cv2.dilate(img,kernel,iterations = 1)

def gaussian_noise(img, mean = 0, var = 4):
    """
    applied gaussian noise to image
    """
    #Gaussian noise
    m,n= img.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(m,n))
    gauss = gauss.reshape(m,n)
    return img + gauss

def salt_pepper(img, p = 0.5, q= 0.004):
    """
    adds salt and pepper nois with quantity q and p percentage between salt and pepper
    """
    #salt and paper 
    row,col = img.shape
    image = np.copy(img)
    # Salt 
    num_salt = np.ceil(q * img.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
        for i in img.shape]
    image[coords] = 1

    # Pepper 
    num_pepper = np.ceil(q* img.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in img.shape]
    image[coords] = 0
    
    return image
