"""
generating captions of photos using deep learning methods

Author : Jitendra kumar chappidi
Libraries used : tensorflow, keras, numpy, nltk

program consists of below sections

01: data preparation/preprocessing
02: understanding the problem using EDA
03: defining,training,evaluation of model
04: generating new predictions

"""


# importing necessary libraries
import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
import numpy as np
from os import listdir, sep
from pickle import dump



epoch = 1
batch_size = 128
input_data_directory = "../../Data/Flicker8k_Dataset"
extracted_features_file = "photo_features_data.pkl"
# checking version
print(f"Tensorflow Version: {tf.__version__} ")
print(f"Keras version: {ks.__version__}")


# 01: data preparation/preprocessing
# extract features from each photo in the directory
def extract_features_photo(input_image_dir):
    # model
    model = VGG16()
    # removing last layer from the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # for storing extracted photo features from VGG16 model
    photo_features_data = dict()
    for file_name in listdir(input_image_dir):
        image_name = input_image_dir + sep + file_name
        # loading image with selected target size from image_file
        image = load_img(image_name, target_size=(224, 224))
        # converting the image to array
        image = img_to_array(image)
        # reshape data for model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # preprocessing image for model
        image = preprocess_input(image)
        # extract features of image from model
        feature = model.predict(image, verbose=0)
        # getting image id from image name
        image_id = file_name.split(".")[0]
        # assigning extracted photo features to photo_id
        photo_features_data[image_id] = feature
        # printing processed image_id from input
        print(f"Processed image: {image_id}")
    return photo_features_data






# 02: understanding the problem using EDA
# 03: defining,training,evaluation of model
# 04: generating new predictions


# calling functions
photo_features_data = extract_features_photo(input_data_directory)
print(f"total extracted features: {len(photo_features_data)}")
# save the extracted features to file
print(f"Saving to file: {extracted_features_file}")
dump(photo_features_data, open(extracted_features_file, "wb"))


