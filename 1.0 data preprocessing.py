# The first file preprocesses the training data set for efficiency, all data set images are converted to grayscale,
# and resized to 100*100 dimensions
# OpenCv,Os,Keras



import cv2,os

data_path='dataset'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories,labels))

print(label_dict)
print(categories)
print(labels)


#Resizing image and converting each image in dataset to grayscale 

img_size=100
data=[]
target=[]


for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
        
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)

        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
            resized=cv2.resize(gray,(img_size,img_size))
            data.append(resized)
            target.append(label_dict[category])  # Image and associated label added to the training dataset 
        
# Exception for error handling images that are unsuccessfully added to dataset, error istnace will be output to user 
        except Exception as e:
            print('Exception:',e)
            


import numpy as np 

data=np.array(data)/255.0  # normalize data set by dividing array by pixel dynamic range (255) 
data=np.reshape(data,(data.shape[0],img_size,img_size,1))  #converting dataset into a 4 dimensional array ( the format required by our convolutional neural network) 
target=np.array(target)

# pass processed training arrays and save as numpy files with (Data,target) labels  
from keras.utils import np_utils
new_target=np_utils.to_categorical(target)
np.save('data',data)
np.save('target',new_target)

