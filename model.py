import csv
import cv2
import sys
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Model

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

# Generator function to increase the efficiency of the training process
def generator(samples, batch_size = 32):
	num_samples = len(samples)

	while  1:
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset + batch_size]

			images = []
			steering_angles = []

			# Extracting images 
			for sample in batch_samples:
				if line[0].startswith("IMG"):
					img_center_path = "/Users/adityaborde/Self-Driving-Car-ND/CarND-Behavioral-Cloning-P3/CarND-Behavioral-Cloning-P3/data/" + str(sample[0])
				else:
					img_center_path = line[0]
				if line[1].startswith(" IMG"):
					img_left_path =   "/Users/adityaborde/Self-Driving-Car-ND/CarND-Behavioral-Cloning-P3/CarND-Behavioral-Cloning-P3/data/" + str(sample[1])
				else:
					img_left_path = line[1]
				if line[2].startswith(" IMG"):
					img_right_path =  "/Users/adityaborde/Self-Driving-Car-ND/CarND-Behavioral-Cloning-P3/CarND-Behavioral-Cloning-P3/data/" + str(sample[2])
				else:
					img_right_path = line[2]	

				# Extracting images from all the angles i.e. center, left, right
				img_center = cv2.imread(img_center_path.replace(" ", ""))
				img_left = cv2.imread(img_left_path.replace(" ", ""))
				img_right = cv2.imread(img_right_path.replace(" ", ""))

				img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2YUV)
				img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2YUV)
				img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2YUV)
				
				# Adding image to the list of images	
				images.append(img_center)
				images.append(img_left)
				images.append(img_right)


				steering_angles.append((float)(sample[3]))
				
				# Adjusting steering angle for left and right images
				steering_angles.append((float)(sample[3]) + 0.2)
				steering_angles.append((float)(sample[3]) - 0.2)

			x_train = np.array(images)
			y_train = np.array(steering_angles)

			# Shuffling the data
			yield shuffle(x_train, y_train)

lines = []

# Reading the csv file containing the image location and steering angles
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)

	for line in reader:
		lines.append(line)

lines = lines[1:]

# Spliting the data into training and validation data
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print('Training Begins...')

# Defining the model for training
model = Sequential()

# Building Network

# Layer 1: Normalization of Image
model.add(Lambda(lambda x: (x / 127.5) - 1, input_shape=(160, 320, 3)))

# Layer 2: Cropping the image so that the relevant portion is highlighted
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

# Layer 3: Convolutional Layer 5x5 kernel
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))

# Layer 4: Convolutional Layer 5x5 kernel
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))

# Layer 5: Convolutional Layer 5x5 kernel
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))

# Layer 6: Convolutional Layer 3x3 kernel
model.add(Convolution2D(64, 3, 3, activation='relu'))

# Layer 7: Convolutional Layer 3x3 kernel
model.add(Convolution2D(64, 3, 3, activation='relu'))

# Flattening image
model.add(Flatten())

# Layer 8: Fully Connected Layer,  Output: 100
model.add(Dense(100))

# Layer 9: Fully Connected Layer,  Output: 50
model.add(Dense(50))

# Layer 10: Fully Connected Layer,  Output: 10
model.add(Dense(10))

# Layer 11: Fully Connected Layer,  Output: 1
model.add(Dense(1))

# Compiling model using MSE as loss function and Adam Optimizer
model.compile(loss='mse', optimizer='adam')

# Providing data to fit for the model
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples=len(validation_samples),  nb_epoch=5)

# Saving model
model.save('model.h5')



