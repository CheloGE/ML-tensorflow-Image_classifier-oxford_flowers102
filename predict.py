import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import logging
import argparse
import requests
import json
from PIL import Image
from io import BytesIO

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser ()
parser.add_argument ('--image_url', default='https://github.com/CheloGE/ML-tensorflow-Image_classifier-oxford_flowers102/blob/master/test_images/hard-leaved_pocket_orchid.jpg?raw=1',
 help = 'Path to image.', type = str)
parser.add_argument('--checkpoint', default='./saved_models/checkpoint_15_05_2020__02_16_19/', help='Point to checkpoint file as str.', type=str)
parser.add_argument ('--top_k', default = 5, help = 'Top K most likely classes.', type = int)
parser.add_argument ('--category_names' , default = 'https://github.com/CheloGE/ML-tensorflow-Image_classifier-oxford_flowers102/blob/master/label_map.json?raw=1', 
help = 'Mapping of categories to real names.', type = str)
# creating object parser and setting parameters
commands = parser.parse_args()
path_to_image = commands.image_url
saved_model_path = commands.checkpoint
k_top = commands.top_k
json_url = commands.category_names
# reloading model
reloaded_model = tf.keras.models.load_model(saved_model_path)
# Create the process_image function
r = requests.get(json_url, stream=True)
class_names=json.loads(r.content.decode('utf-8'))
class_names = {int(k):v for k,v in class_names.items()}
## defining predictions
def resize_image(image, size=(224,224)):
    """
      This functions gets a bunch of images and resize them to the input size
      @params size: desired size of the images
      @params images: a tensor group of images to resize

      @return resized_images: a numpy array with all new resized images
    """
    return tf.image.resize(image, size)
def normalize_image(images, a=-1, b=1, minPix=[0], maxPix=[255]):
    """
    Normalize the image data with Min-Max scaling to a range of [a, b] 
    @params images: a tensor of images data to be normalized
    @return: tensor of Normalized image data
    """
    a = tf.constant([a], dtype=tf.float32)
    b = tf.constant([b], dtype=tf.float32)
    min_pixel = tf.constant(minPix, dtype=tf.float32)
    max_pixel = tf.constant(maxPix, dtype=tf.float32)

    return a + (((images - min_pixel)*(b - a) )/(max_pixel - min_pixel))
def process_image(image):
    """
        This function preprocess and image
    """
    image = tf.convert_to_tensor(image, np.float32)
    feature = normalize_image(resize_image(image))
    return feature
# Create the predict function
def predict(image_path, model, top_k):
    response = requests.get(image_path, stream=True)
    im = Image.open(BytesIO(response.content))
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    prediction = model.predict(np.expand_dims(processed_test_image, axis=0))
    top_values, top_indices = tf.math.top_k(prediction, top_k)
    top_classes = [class_names[value+1] for value in top_indices.cpu().numpy()[0]]
    ## Plotting images
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1,2,1)
    plt.imshow(normalize_image(processed_test_image, 0, 1, [-1],[1]))
    plt.title(f"Predicted label: {top_classes[0]}")
    ax = fig.add_subplot(1,2,2)
    plt.barh(top_classes, top_values.numpy()[0]*100)
    fig.tight_layout(w_pad=5)
    return top_values.numpy()[0], top_classes
probs, classes = predict(path_to_image, reloaded_model, k_top)
print(f"top {k_top} probabilities for the image are: {probs}")
print(f"top {k_top} classes for the image are: {classes}")