#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras 
from matplotlib import pyplot as plt
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D


# In[ ]:


trainx = pd.read_csv('../input/ahcd1/csvTrainImages 13440x1024.csv', header = None)
trainy = pd.read_csv('../input/ahcd1/csvTrainLabel 13440x1.csv', header = None)
testx = pd.read_csv('../input/ahcd1/csvTestImages 3360x1024.csv', header = None)
testy = pd.read_csv('../input/ahcd1/csvTestLabel 3360x1.csv', header = None)


# In[ ]:


trainx = trainx.values.astype('float32')
#training labels
trainy = trainy.values.astype('int32')-1

#testing images
testx = testx.values.astype('float32')
#testing labels
testy = testy.values.astype('int32')-1


# In[ ]:


trainx[0]


# In[ ]:


trainy = keras.utils.to_categorical(trainy,28)


# In[ ]:


print(trainy.shape)


# In[ ]:


testy = keras.utils.to_categorical(testy,28)


# In[ ]:


trainx = trainx.reshape([-1, 32, 32,1])
testx = testx.reshape([-1, 32, 32,1])


# In[ ]:


print(trainx.shape, trainy.shape, testx.shape, testy.shape)


# In[ ]:


#Normalising data
trainx /=255.0
testx /=255.0
trainx.shape


# In[ ]:


recognizer = Sequential()

recognizer.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (32,32,1)))
recognizer.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
recognizer.add(MaxPool2D(pool_size=(2,2)))
recognizer.add(Dropout(0.25))
recognizer.add(Flatten())
recognizer.add(Dense(28, activation = "softmax"))


# In[ ]:


recognizer.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


recognizer.fit(trainx, trainy,batch_size=100, epochs=50,validation_data=(testx,testy))


# In[ ]:


#plotting 
testx = testx.reshape([-1, 32, 32])
testx.shape
plt.figure()
plt.imshow(testx[50].T)
plt.colorbar()
plt.grid(False)
plt.show()


# In[ ]:


#predcting Model
img = testx[100]

print(img.shape)

img = (np.expand_dims(img,0))

print(img.shape)


# In[ ]:


predictions_single = model.predict(img)

print(predictions_single)


# In[ ]:


i=np.argmax(predictions_single[0])


# In[ ]:


catego = ["alif","ba2","ta2","thaa","jim","7a2","kha2","dal","dhal","ra2","zay","sin","shin","saad","daad","ta2on","zaa","3ayn","ghayn","faa","qaaf","kaf","lam","mim","noon","haa","waw","yaa"]


# In[ ]:


print(catego[i])


# In[ ]:


# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(testx, testy, batch_size=128)
print("test loss, test acc:", results)

