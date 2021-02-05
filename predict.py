
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plot
import logging 
import ImageDataGenerator
import time
import json
import argparse
import sys
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from tensorflow.keras.preprocessing.image 
from module import sub-module


%matplotlib inline
%config InlineBackend.figure_format = 'retina'
 
parser = argparse.ArgumentParser()
 


parser.add_argument('-checkpoint',help='Checkpoint of the model')
parser.add_argument('--gpu', action='store_true',  default=False,dest='gpu', help='Set training to gpu')
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('-image_dir', default='./test_image/hard-leaved_pocket_orchid.jpg',help='Path to image,',type=str)
parser.add_argument('--category_names', action='store',dest='category_names', default=None,help='Json association between category and names')
parser.add_argument('image_path',help='Image directory path')
parser.add_argument('image_path', help='image  path')


 

commands = parser.parse_args()
#top_k = commands.top_k
top_k = commands.topk
image_path = commands.checkpoint
classes = commands.category_names
commands , unknown = parser.parse_known_args()

reload_keras_model = tf.keras.models.load_model( commands.checkpoint, custom_objects={'KerasLayer':hub.KerasLayer})
#reload_keras_model = tf.keras.models.load_model("./mymodel.h5",custom_objects = {'KerasLayer':hub.kerasLayer})
with open('label_map.json', 'r') as f:class_names = json.load(f)
print(class_names)

def process_image(image):
    image_size = 224
    image = tf.convert_to_tensor(image)
    image = tf.cast(image,tf.float32)
    image = tf.image.resize(image,(image_size,image_size))
    image /= 255
    image = image.numpy()
    return image
	
def predict(image_path, model, top_k):
    imag = Image.open(image_path)
    np_image = np.asarray(imag)
    modify_np_image =process_image(np_image)
    modify_np_image = np.expand_dims(modify_np_image, axis=0)
    prediction = model.predict(modify_np_image)
    probs, classes = tf.math.top_k(prediction,top_k)
	return probs.numpy()[0], classes.numpy()[0]
	
probs, class_names = predict(image_path , reload_keras_model, top_k)

