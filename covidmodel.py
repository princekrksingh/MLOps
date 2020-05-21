#!/usr/bin/env python
# coding: utf-8

# # <font color ='purple'>Let's start by Importing the required libraries </font>

# In[1]:



import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
from PIL import Image
print(os.listdir("chest_xray"))
import tensorflow as tf
# Keras Libraries


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, load_img


# In[2]:


mainDIR = os.listdir('chest_xray/chest_xray')
print(mainDIR)


# In[3]:


train_folder= 'chest_xray/chest_xray/train/'
val_folder = 'chest_xray/chest_xray/val/'
test_folder = 'chest_xray/chest_xray/test/'


# ## Let's set up the training and testing folders.
# 

# In[4]:


# train 
os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'covid/'


# ## Let's take a look at some of the pictures.
# 
# 

# In[5]:


#Normal pic 
print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ',norm_pic)

norm_pic_address = train_n+norm_pic

#covid
rand_p = np.random.randint(0,len(os.listdir(train_p)))

sic_pic =  os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('covid picture title:', sic_pic)

# Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

#Let's plt these images
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('covid')


# In[6]:


# let's build the CNN model

cnn = Sequential()

#Convolution
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))

#Pooling
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# 2nd Convolution
cnn.add(Conv2D(32, (3, 3), activation="relu"))

# 2nd Pooling layer
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# Flatten the layer
cnn.add(Flatten())

# Fully Connected Layers
cnn.add(Dense(activation = 'relu', units = 128))
cnn.add(Dense(activation = 'sigmoid', units = 1))

# Compile the Neural network
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])





# # <font color='purple'>Now, we are going to fit the model to our training dataset and we will keep out testing dataset seperate </font>

# In[7]:


# Fitting the CNN to the images
# The function ImageDataGenerator augments your image by iterating through image as your CNN is getting ready to process that image

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)  #Image normalization.

training_set = train_datagen.flow_from_directory('chest_xray/chest_xray/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('chest_xray/chest_xray/val/',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory('chest_xray/chest_xray/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# # This summary is a great way for us to see how our CNN is being set up

# In[8]:


cnn.summary()


# In[10]:


cnn_model = cnn.fit_generator(training_set,
                         steps_per_epoch = 50,
                         epochs = 2,
                         validation_data = validation_generator,
                         validation_steps = 624)


# In[11]:


test_accu = cnn.evaluate_generator(test_set,steps=624)


# In[12]:


print('The testing accuracy is :',test_accu[1]*100, '%')


# In[13]:


# Accuracy 
plt.plot(cnn_model.history['accuracy'])
plt.plot(cnn_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()


# In[14]:


# Loss 

plt.plot(cnn_model.history['val_loss'])
plt.plot(cnn_model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show()


# In[25]:


from keras.models import load_model

cnn.save('myCovid_model.h5')  


# In[26]:


from keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)


# In[27]:


from keras.models import load_model
model = load_model('myCovid_model.h5')
print(model.outputs)
print(model.inputs)


# In[ ]:


cnn_model.predict("")

