from keras.layers import Conv2D,MaxPooling2D
from keras.layers import BatchNormalization
#from keras.layers.merge import add,concatenate
from keras.models import Sequential
#to_categorical
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import cv2


train_path = 'D:/ml/train'
train_dog = ['D:/ml/train/{}'.format(i) for i in os.listdir(train_path) if 'dog' in i]
train_cat = ['D:/ml/train/{}'.format(i) for i in os.listdir(train_path) if 'cat' in i]
#test_images = ['D:/ml/test1/test1 {}'.format(i) for i in os.listdir(test_path)]
train_images = train_dog[:200] + train_cat[:200]
random.shuffle(train_images)
rows = 400
columns = 400
channels = 1



def resize_image_and_attched_labels(list_of_images):
    X = []
    y = []
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image,0), 
                            (rows,columns),
                            interpolation = cv2.INTER_CUBIC))
        if 'dog' in image:
            y.append(1)
        elif 'cat' in image:
            y.append(0)
    return X, y
X, y = resize_image_and_attched_labels(train_images)
X = np.array(X)
y = np.array(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state = 2)
ntrain = len(X_train)
nval = len(X_val)

batch_size = 32
y_train = to_categorical(y_train)      
y_val = to_categorical(y_val)
X_train = X_train.reshape(ntrain,400,400,1)
X_val = X_val.reshape(nval,400,400,1)
      
X_val = to_categorical(X_val)


model = Sequential()
model.add(Conv2D(64,3,3, activation='relu',input_shape=(400,400,1)))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(128,3,3,activation='relu'))
model.add(Conv2D(128,3,3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(256,3,3,activation='relu'))
model.add(Conv2D(256,3,3,activation='relu'))
model.add(Conv2D(256,3,3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(512,3,3,activation='relu'))
model.add(Conv2D(512,3,3,activation='relu'))
model.add(Conv2D(512,3,3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(512,3,3,activation='relu'))
model.add(Conv2D(512,3,3,activation='relu'))
model.add(Conv2D(512,3,3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=1))
u = model.add(Conv2D(512,3,3,activation='relu'))
model.add(Conv2D(1024,3,3,activation='relu'))
v = model.add(Conv2D(1024,1,1,activation='relu')) 
model.add(Conv2D(256,1,1,activation='relu'))
w = model.add(Conv2D(512,(3,3),activation='relu'))
model.add(Conv2D(128,1,1,activation='relu'))
x = model.add(Conv2D(256,(3,3),activation='relu'))#remove stride 2
model.add(Conv2D(128,1,1,activation='relu'))
y = model.add(Conv2D(256,(3,3),activation='relu'))#remove stride 2
model.add(Conv2D(128,1,1,activation='relu'))
z = model.add(Conv2D(256,(3,3),activation='relu'))#remove stride 2

model.compile(
        optimizer = 'Adam',
        loss = 'categorical_crossentropy'
        )
model.fit(
        X_train,
        y_train,
        epochs = 10,
        batch_size = batch_size
        )
model.save('model_ssd.h5')


    
    
    