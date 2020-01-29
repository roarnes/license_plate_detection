import cv2
import numpy as np

from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

classifier = Sequential()

classifier.add(Conv2D(32, (3,3), input_shape = (64,64,1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())

classifier.add(Dense(units = 100, activation = 'relu'))
classifier.add(Dense(units = 120, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator (rescale = 1./255, 
									shear_range = 0.2, 
									zoom_range = 0.2, 
									horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('new_training_data_canny/', 
												target_size = (64, 64), 
												color_mode = 'grayscale',
												batch_size = 32, 
												class_mode = 'binary')

test_set = test_datagen.flow_from_directory('new_test_data_canny/', 
											target_size = (64, 64), 
											color_mode = 'grayscale',
											batch_size = 32, 
											class_mode = 'binary')

classifier.fit_generator(training_set, 
						steps_per_epoch = 100, 
						epochs = 3, 
						validation_data = test_set, 
						validation_steps = 10)


img = cv2.imread('full1.png')
edges = cv2.Canny(img, 100, 200)

stepSize = 64
(w_width, w_height) = (64, 64)

onceFlag = 0

for x in range(0, edges.shape[1] - w_width, stepSize):
	for y in range(0, edges.shape[0] - w_height, stepSize):
		window = edges[x:x + w_width, y:y + w_height]

		if window.shape != (64, 64):
			break
      
		# classify content of the window with your classifier and  
		# determine if the window includes an object (cell) or not
		test_image = window
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis = 0)
		result = classifier.predict(test_image)
		training_set.class_indices
		if result[0][0] == 1:
			prediction = 'gaadaplat'
		else:
			prediction = 'adaplat'

		print(prediction)

		if onceFlag == 0 and prediction == 'adaplat':
			cv2.rectangle(img, (x, y), (x + w_width, y + w_height), (255, 0, 0), 2) # draw rectangle on image
			plt.imshow(np.array(img).astype('uint8'))
		
plt.show()