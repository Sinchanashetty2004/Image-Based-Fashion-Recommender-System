import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Load the pretrained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add global max pooling layer
model = tensorflow.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Folder containing the images
image_folder = 'images'

# Get valid image file paths
filenames = []
for file in os.listdir(image_folder):
    full_path = os.path.join(image_folder, file)
    if os.path.isfile(full_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
        filenames.append(full_path)

# Extract features and store in a list
feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

# Save features and filenames using pickle
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
