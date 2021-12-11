from tensorflow.python.keras.datasets import cifar10
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.layers \
    import Dense,Conv2D,MaxPool2D,Flatten
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping

(x_train , y_train) , (x_test , y_test)  = cifar10.load_data()

x_train = x_train/255
x_test = x_test/255

y_train_cat = to_categorical(y_train,10)
y_test_cat = to_categorical(y_test,10)

early_stop = EarlyStopping(monitor='val_loss' , mode='min' ,
                           verbose=1)
model = Sequential()
model.add(Conv2D(filters=32 , kernel_size=(4,4)
                 , input_shape=(32,32,3) , activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32 , kernel_size=(4,4)
                 , input_shape=(32,32,3) , activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256 , activation='relu'))
model.add(Dense(10 , activation='softmax'))

model.compile(loss='categorical_crossentropy' , optimizer='adam'
              ,metrics=['accuracy'])
model.fit(x_train , y_train_cat , epochs=100 ,
validation_data=(x_test,y_test_cat) , callbacks=[early_stop])

#%%
labels =[ 'airplane' ,'automobile' ,'bird', 'cat' ,'deer' ,'dog'
    ,'frog' ,'horse']
from sklearn.metrics import confusion_matrix,classification_report
prediction = model.predict_classes(x_test)
classification_report(y_test,prediction)
#%%
plt.imshow(x_test[0])
plt.title(labels[prediction[0]])