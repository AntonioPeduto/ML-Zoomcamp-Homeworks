#!/usr/bin/env python
# coding: utf-8

# Import libraries
import tflite_runtime.interpreter as tflite

import tensorflow as tf
import numpy as np
import os

from io import BytesIO
from urllib import request
from PIL import Image


model_tflite='bees-wasps-v2.tflite'
# Load the TFLite model in TFLite Interpreter and allocate tensors
interpreter = tf.lite.Interpreter(model_path=os.getcwd()+'/'+model_tflite)
interpreter.allocate_tensors()


# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Get input and output index
input_index = input_details[0]['index']
output_index = output_details[0]['index']

# Get input size of model TFLite
input_size_model = input_details[0]['shape'].tolist()
# Set new size of image
target_size = input_size_model[1:3]

# Definition of classes 
classes = ['wasps','bees']

# Preparing the image
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def predict(url):
    # Download image
    img = download_image(url)
    # Resize image
    img = prepare_image(img, target_size)
    # Get image as array
    img_array = np.array(img)
    # Pre-processing of image
    img_array_preprocess = img_array.reshape(1,target_size[0],target_size[1],3)
    img_array_preprocess = img_array_preprocess/255.
    # Define input data
    input_data = img_array_preprocess.astype(np.float32)
    # Predict output
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_index)
    floating_output = output_data[0].tolist()
    floating_prediction = [floating_output[0], 1. - floating_output[0]]
    return dict(zip(classes, floating_prediction))

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

