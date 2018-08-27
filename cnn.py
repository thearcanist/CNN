#importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#creating object and Passing Neural network type....Initialising CNN
classifier = Sequential()

#Step 1. Adding Convolution step .... Conv2D takes 4 arguments 1st is number of filters 2nd is the shape of the filter, 3rd is the input shape (dimensions) and the type of each input image.4th argument is Activation function type - like sigmoid or relu
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#Step 2. Now we do the pooling operation to reduce the size of the images. 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adding a second Convolution layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3. Converting 2D image to 1D i.e pooled image crystals and converting them into 1D
classifier.add(Flatten())

#step 4. Full Connection.... Connecting all the Nodes in the CNN
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Part 2 - Fitting in the CNN to the image

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

train_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directry('dataset/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

test_set = train_datagen.flow_from_directry('dataset/test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

classifier.fitgenerator(training_set, steps_per_epoch = 8000, epochs = 25, validation_data = test_set, validation_steps = 2000)

#Part 3 - Make Predictions 
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64,64));
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(est_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

    
print (prediction)
