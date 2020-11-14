# Design of our convolutional Neural network architecture, data was trained with 20 epochs.
#After running the Epoch with the lowest validation loss and highest accuracy was selected to be 
#our model for image classification in the final stage.

#loading previously saved data 
import numpy as np
data=np.load('data.npy')
target=np.load('target.npy')


from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint

#CNN Structre :  200-> 100 -> Flattened -> Dense 50 neurons -> Dense 2 Neurons-> 0:Mask || 1: NO MASK 

model=Sequential()

#First Convolutional Neural network layer (200)_  Layer sublayers(2D Convolution-> Relu layer -> 2D Maxpooling)
model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Second CNN layer (100)
model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Flatten colvolutional output converge into 50 and 2 dense neurons respectively, to a final layer with two ouputs
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



# process and output  validation loss, accuracy, validation accuracy for each epoch, Model belonging to the 
# epoch with the lowest Validation loss and highest accuracy is selected  

from sklearn.model_selection import train_test_split
train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)

checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)


# Plot of validation loss VS Training loss for each Epoch 


from matplotlib import pyplot as plt

plt.plot(history.history['loss'],'r',label='Training loss')
plt.plot(history.history['val_loss'],label='Validation loss')
plt.title('Losses recorded for each Epoch ')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Plot of validation loss VS Training loss for each Epoch 


plt.plot(history.history['accuracy'],'r',label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Print statistics for each epoch

print(model.evaluate(test_data,test_target))

