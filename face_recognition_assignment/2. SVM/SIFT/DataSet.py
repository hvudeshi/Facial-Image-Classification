import os
import math
import random

"""
   Function: LabelData 
   Input: Name of the database, either ORL or Shuffield, note that both 
   Output: Train set and test set
   
   Description: The function retreive the images from each folder, labled the images of the same person with the same label, 
                split the images randomly to train and test set.
   
"""
def LabelData(dataset):
    dataDir = "../Dataset/"+dataset+"/"
    listDir = os.listdir(dataDir)
    train_set = []
    test_set = []
    classId = 0
    
    for individuDir in listDir: 
        
        train_img = os.listdir(dataDir+ individuDir+"/train/")
        test_img = os.listdir(dataDir+ individuDir+"/test/")
        
        
        #t_size = math.floor(len(list_img)*0.6)
        #random.shuffle(list_img)
        # print(individuDir)
        # print("\n")


        for img in train_img:
                
            #tup = (dataDir+ individuDir+"/" + img, classId)
            print(img)
            print("\n")
            tup = (dataDir+ individuDir+"/train/" + img, individuDir)
            #if t_size >0:
            train_set.append(tup)
            #    t_size -= 1
            #else:
            
        #classId += 1 
        
        for img in test_img:
            
            print(img)
            print("\n")
            tup = (dataDir+ individuDir+"/test/" + img, individuDir)
            test_set.append(tup)
            
    return train_set, test_set 
LabelData("orl")
