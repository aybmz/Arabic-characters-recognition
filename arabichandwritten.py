#importing libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from matplotlib import pyplot as plt
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
#loading data
trainx = pd.read_csv('../input/ahcd1/csvTrainImages 13440x1024.csv', header = None)
trainy = pd.read_csv('../input/ahcd1/csvTrainLabel 13440x1.csv', header = None)
testx = pd.read_csv('../input/ahcd1/csvTestImages 3360x1024.csv', header = None)
testy = pd.read_csv('../input/ahcd1/csvTestLabel 3360x1.csv', header = None)
trainx = trainx.values.astype('float32')
#training labels
trainy = trainy.values.astype('int32')-1

#testing images
testx = testx.values.astype('float32')
#testing labels
testy = testy.values.astype('int32')-1
trainy = keras.utils.to_categorical(trainy,28)
print(trainy.shape)
testy = keras.utils.to_categorical(testy,28)
#reshaping 
trainx = trainx.reshape([-1, 32, 32,1])
testx = testx.reshape([-1, 32, 32,1])
print(trainx.shape, trainy.shape, testx.shape, testy.shape)
#Normalising data
trainx /=255.0
testx /=255.0
trainx.shape
#model CNN
recognizer = Sequential()

recognizer.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (32,32,1)))
recognizer.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
recognizer.add(MaxPool2D(pool_size=(2,2)))
recognizer.add(Dropout(0.25))
recognizer.add(Flatten())
recognizer.add(Dense(28, activation = "softmax"))
#compiling the model
recognizer.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

recognizer.fit(trainx, trainy,batch_size=100, epochs=50,validation_data=(testx,testy))
#visualising data
testx = testx.reshape([-1, 32, 32]) # reshaping  (2D array)
testx.shape
plt.figure()
plt.imshow(testx[50].T)
plt.colorbar()
plt.grid(False)
plt.show()
testx = testx.reshape([-1,32,32,1])
img = testx[50]
print(img.shape)
img = (np.expand_dims(img,0))
print(img.shape)
#predcting Model
predictions_single = recognizer.predict(img)
print(predictions_single)
i=np.argmax(predictions_single[0])
labelsArabic = ["alif","ba2","ta2","thaa","jim","7a2","kha2","dal","dhal","ra2","zay","sin","shin","saad","daad","ta2on","zaa","3ayn","ghayn","faa","qaaf","kaf","lam","mim","noon","haa","waw","yaa"]
print(labelsArabic[i])
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = recognizer.evaluate(testx, testy, batch_size=128)
print("test loss, test acc:", results)

