# Video Capture, Classification and labeling 

from keras.models import load_model
import cv2
import numpy as np

# Model from epoch 17 selected as it has the lowest validation loss and highest accuracy in contrast to the other epochs
#Video capture frame activated

model = load_model('model-017.model')
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(0)

# Binary label assigned to frame, No mask labeled with red and with mask labeled with green
labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)} 


# Loop will capture each frame, extract region of interest (face),
# Process the ROI in the same way it was done in training (convert to grayscale and resize) 
# Processed image is normalized and fed into the imported Neural model for prediction (0 :Mask | 1: No Mask) 

while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for x,y,w,h in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
      

  #Draw a rectangle on the frame over the ROI with the approriate color obtained from dict ( Green= Mask , red=no mask )
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) # add Binary label on display
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()

