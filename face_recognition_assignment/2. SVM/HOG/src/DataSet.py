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
    data = []
    classId = 0
    
    for individuDir in listDir: 
        
        list_img = os.listdir(dataDir+individuDir)
        #t_size = math.floor(len(list_img)*0.6)
        random.shuffle(list_img)
#        print(individuDir)
        #print("\n")


        for img in list_img:
                
            #tup = (dataDir+ individuDir+"/" + img, classId)
#            print(img)
#            print("\n")
            tup = (dataDir+ individuDir+"/" + img, individuDir)
            #if t_size >0:
            data.append(tup)
            #    t_size -= 1
            #else:
            
        #classId += 1 
            
    return data
LabelData("orl")
