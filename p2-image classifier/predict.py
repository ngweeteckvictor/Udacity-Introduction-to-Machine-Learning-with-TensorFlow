import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image

#the process image function
def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, [224, 224])
    image /= 255
    return image.numpy()

#the predict function
def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)
    
    probs = model.predict(image)[0]
    classes = np.argsort(-probs)[:top_k]
    
    return probs[classes[:]], classes + 1

#Parse the command inputs
parser = argparse.ArgumentParser(description='Predict the top flower names from an image with their corresponding probabilities')
parser.add_argument('image_path', help='Path to a input image')
parser.add_argument('model', help='Path to a trained model')
parser.add_argument('--top_k', type=int, default=1, help='Return the top K most likely classes')
parser.add_argument('--category_names', help='Path to a JSON mapping file')
args = parser.parse_args()

#load the trained model
my_model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer':hub.KerasLayer})

#predict the flower from the input image
probs, classes = predict(args.image_path, my_model, args.top_k)

#path to a JSON mapping file is provided
if(args.category_names):
    with open(args.category_names) as f:
        class_names = json.load(f)

    flower_names = []

    for label in classes:
        flower_names.append(class_names[str(label)])

    #display flower names with their probabilities
    for x in range(len(flower_names)):
        print('Flower name: {} Probability: {:.3%}'.format(flower_names[x], probs[x]))

else:
    #display flower classes with their probabilities
    for x in range(classes.size):
        print('Flower Class: {} Probability: {:.3%}'.format(classes[x], probs[x]))