# Behavioural-Cloning

A project to emmulate the driving recorded using the simulator provided by Udacity as part of its Nanodegree Program

# Table of Content

1. model.py
- Used to train the deep learning model using the images an steering angle recorded during the simulation of driving over the provided track

2. drive.py
- Used to drive the car over the provided track using the model and the parameters saved in model.h5 file

3. model.h5
- The file generated after training the model

# Requirements
- Python
- Keras
- Numpy
- Sklearn
- Other supporting imports mentioned in the respective .py files

# Usage
- Generate dataset using the simulator provided by Udacity
- Use model.py to train over the dataset. [Edit the path to fetch the dataset in model.py according to the location of dataset on your local machine]
- model.h5 file will be generated.
- Use drive.py to drive the car on the track using model.h5 as a command line argument to drive.py file

# License
- Included in the repo
