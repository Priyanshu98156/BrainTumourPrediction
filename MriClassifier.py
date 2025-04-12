#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import matplotlib.pyplot as plt 
import cv2
import numpy as np  
import pandas as pd 
from sklearn.model_selection import train_test_split
import tensorflow as tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers 


# In[2]:


dataset_path = "Datasets/Mri_classifier"
categories  = ["Mri","Not_Mri"]
data = []
labels = []
image_size = 128


# In[3]:


type(data)


# In[4]:


for category in categories:
    
    folder_path = os.path.join(dataset_path,category)
    label = categories.index(category)
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path,image_name)

        #read the image in grayscale
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue
        else:
            # Resize the image to the target size
            image = cv2.resize(image, (image_size, image_size))
            # Normalize pixel values to range [0, 1]
            image = image / 255.0
        data.append(image)
        labels.append(label)


# In[5]:


data = np.array(data).reshape(-1,image_size,image_size,1)
labels = np.array(labels)
X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size= 0.02,random_state=42)


# In[16]:


batch_size = 32

datagen = ImageDataGenerator(
    rotation_range= 0,
    shear_range=0,        # Shear transformation
    zoom_range=0,         # Random zoom
    horizontal_flip=False,   # Flip images horizontally
    # fill_mode='nearest'
)
val_datagen = ImageDataGenerator()
val_generator = val_datagen.flow(X_test,y_test , batch_size = batch_size)


# In[17]:


datagen.fit(X_train)


# In[18]:


model = Sequential()

# 🧱 Block 1
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D((2, 2)))

# 🧱 Block 2
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# 🧱 Block 3
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# 🧠 Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

model.summary()


# In[19]:


optimizer = RMSprop(learning_rate=0.01)


# In[20]:


model.compile(optimizer= optimizer,loss = keras.losses.binary_crossentropy,metrics=["accuracy"])


# In[27]:


from keras.callbacks import ModelCheckpoint,EarlyStopping

es = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.01,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0,
)
mc = ModelCheckpoint(
    filepath = "./bestmodelmriclassifier.h5",
    monitor="val_accuracy",
    verbose=0,
    save_best_only=True,
    mode="auto",
    
)
cd = [es,mc]


# In[28]:


batch_size = 32

train_generator = datagen.flow(X_train,y_train,batch_size = batch_size)

history = model.fit(train_generator, epochs=10, validation_data=(X_test, y_test),callbacks = cd)


# In[29]:


test_looss,test_acc = model.evaluate(X_test,y_test)
print(f"Test Accuracy :{test_acc*100:.2f}%")


# In[31]:


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()


# In[25]:


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()


# In[ ]:




