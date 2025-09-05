import pandas as pd
import numpy as np
import os
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from PIL import ImageOps
# Load the trained model
model_gender = load_model('E:\GENDERUFTK\gen_beta_s4.h5')

# Define the directory containing your images
image_dir = r"E:\GENDERUFTK\cropfacerSON"  # Replace with the path to your image directory

# Function to process and predict an image
def process_and_predict(file):
    im = Image.open(file)
    im = ImageOps.fit(im, (224, 224), Image.LANCZOS)
    im = im.resize((224, 224))
    width, height = im.size
    if width == height:
        im = im.resize((224, 224), Image.BILINEAR)
    else:
        if width > height:
            left = width/2 - height/2
            right = width/2 + height/2
            top = 0
            bottom = height
            im = im.crop((left, top, right, bottom))
            im = im.resize((224, 224), Image.BILINEAR)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            im = im.crop((left, top, right, bottom))
            im = im.resize((224, 224), Image.BILINEAR)
            
    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(-1, 224, 224, 3)
    
    gender_prob = model_gender.predict(ar)[0]
    
    return gender_prob

image_files = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(('.jpg', '.jpeg', '.png'))]
threshold_values = []
total_images = len(image_files)
for image_file in image_files:
    gender_prob = process_and_predict(image_file)
    threshold_values.append(gender_prob)
average_threshold = sum(threshold_values) / total_images
print("Average Threshold:", average_threshold)
# from PIL import Image, ImageOps
# import numpy as np
# import os
# from tensorflow.keras.models import load_model

# # Load the trained model with the correct input shape
# model_gender = load_model('E:\GENDERUFTK\gen_beta_s3.h5', compile=False)

# # Assuming model_gender is loaded with compile=False
# target_size = (224, 224)

# # Function to add padding, resize, and predict gender
# def process_and_predict(file):
#     # Open the image
#     im = Image.open(file)

#     # Resize the image to (224, 224)
#     im = im.resize(target_size, Image.BILINEAR)

#     # Convert the image to a numpy array
#     ar = np.asarray(im)

#     # Normalize the pixel values
#     ar = ar.astype('float32') / 255.0

#     # Reshape the array for model input
#     ar = ar.reshape(-1, target_size[0], target_size[1], 3)

#     # Predict gender using the model
#     gender_prob = model_gender.predict(ar)[0]

#     return gender_prob

# # Define the directory containing your images
# image_dir = r"E:\GENDERUFTK\cropfacerSON"

# # Process images in the specified directory
# image_files = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(('.jpg', '.jpeg', '.png'))]
# threshold_values = []

# # Iterate through each image
# for image_file in image_files:
#     gender_prob = process_and_predict(image_file)
#     threshold_values.append(gender_prob)

# # Calculate the average threshold
# average_threshold = sum(threshold_values) / len(threshold_values)
# print("Average Threshold:", average_threshold)
