import numpy as np
from imutils import perspective
from imutils import contours
import imutils
#import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import img_to_array , load_img
from sklearn.model_selection import train_test_split
import os
import random
from keras.utils import to_categorical
#from PIL import Image

test_path = 'D:/ml/test1/test1'

train_path = 'D:/ml/train'
train_dog = ['D:/ml/train/{}'.format(i) for i in os.listdir(train_path) if 'dog' in i]
train_cat = ['D:/ml/train/{}'.format(i) for i in os.listdir(train_path) if 'cat' in i]
test_images = ['D:/ml/test1/test1 {}'.format(i) for i in os.listdir(test_path)]
train_images = train_dog[:9000] + train_cat[:9000]
random.shuffle(train_images)

rows = 150
columns = 150
channels = 1
'''
test_image = cv2.resize(cv2.imread("C:/Users/hp/Downloads/28.jpg",0), 
                            (rows,columns),
                            interpolation = cv2.INTER_CUBIC)
image = test_image
test_label = [0]
test_label = np.asarray(test_label)
test_label = to_categorical(test_label,2)
test_image = test_image.reshape(1,150,150,1)
'''
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
X_train = X_train.reshape(ntrain,150,150,1)
X_val = X_val.reshape(nval,150,150,1)

batch_size = 32

model = Sequential([
  Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(150,150,1)),
  Conv2D(64, kernel_size=(3, 3)),
  BatchNormalization(),
  MaxPooling2D(pool_size=(2, 2),strides=(2,2)),
  Conv2D(128, (3, 3), activation='relu'),
  Conv2D(128, (3, 3), activation='relu'),
  BatchNormalization(),
  MaxPooling2D(pool_size=(2, 2),strides=(2,2)),
  Conv2D(256, (3, 3), activation='relu'),
  Conv2D(256, (3, 3), activation='relu'),
  Conv2D(256, (3, 3), activation='relu'),
  Conv2D(256, (3, 3), activation='relu'),
  BatchNormalization(),
  MaxPooling2D(pool_size=(2, 2),strides=(2,2)),
  Conv2D(512, (3, 3), activation='relu'),
  Conv2D(512, (3, 3), activation='relu'),
  Conv2D(512, (3, 3), activation='relu'),
  Conv2D(512, (3, 3), activation='relu'),
  Conv2D(512, (3, 3), activation='relu'),
  MaxPooling2D(pool_size=(2, 2),strides=(2,2)),
  BatchNormalization(),
  
  #Flatten(),
  #Dropout(0.75),
  #Dense(1024, activation='relu', input_shape=(784,)),
  #Dense(1024, activation='relu'),
  #Conv2D(512, kernel_size=(3, 3)),
  Dense(2, activation='softmax'),
])


#model.load_weights('model_weights.h5')

model.compile(
  optimizer='Adam',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)
'''
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip= True,)
val_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow(X_train,y_train,batch_size=batch_size)
val_generator = val_datagen.flow(X_val,y_val,batch_size=batch_size)
'''
model.fit(
        X_train,
        y_train,      
        epochs=10,
        batch_size = batch_size,
)

model.save_weights('model_ssd.h5')
model.save('model_vgg19_ssd.h5')

result = model.evaluate(
        X_val,
        y_val
        )
#pred = model.predict(test_image)
'''
def Bounding_box(path):
    images = path
    #blured = cv2.medianBlur(image,15)
    blured = cv2.GaussianBlur(images, (9, 9),cv2.BORDER_DEFAULT)
    
    edged = cv2.Canny(blured,100,100)
    edged = cv2.dilate(edged, None, iterations = 12)
    edged = cv2.erode(edged, None, iterations = 12)
    contour = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    (contour,_) = contours.sort_contours(contour)
    contour = max(contour,key=cv2.contourArea)
    box = cv2.minAreaRect(contour)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    return box   
if pred[0][0]>0.5:
    box = Bounding_box(image)
    point = perspective.order_points(box)
    cv2.drawContours(image,[box.astype("int")], -1, (0,100,0), 4)
    #cv2.putText(image,'It is a cat',point[0],-1,(100,0,0),1)
    cv2.imshow('image',image)
    cv2.waitKey(0)
elif pred[0][1]>0.5:
    box = Bounding_box(image)
    point = perspective.order_points(box)
    cv2.drawContours(image,[box.astype("int")], -1, (0,100,0), 4)
    #cv2.putText(image,'It is a dog',-1,(100,0,0),1)
    cv2.imshow('image',image)
    cv2.waitKey(0)
'''
print('total loss: ',result[0])
print('accuracy: ',result[1])
    
