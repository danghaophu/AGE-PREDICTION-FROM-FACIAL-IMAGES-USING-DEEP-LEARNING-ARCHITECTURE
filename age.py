import pandas as pd
import numpy as np
import seaborn as sns
import os
import PIL
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Dense,Dropout,Conv2DTranspose,concatenate
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from keras.utils import to_categorical
from tensorflow.keras.layers import UpSampling2D

# Define your data path and load images
train_data_path = r"E:\GENDERUFTK\cropfacerSON"
images = []
ages = []
genders = []

for i in os.listdir(train_data_path)[0:8000]:
    split = i.split('_')
    ages.append(int(split[0]))
    genders.append(int(split[1]))
    image_path = os.path.join(train_data_path, i)
    images.append(Image.open(image_path))

# Create data Series
images = pd.Series(list(images), name='Images')
ages_series = pd.Series(ages, name='Ages')
genders_series = pd.Series(genders, name='Genders')
data = pd.concat([images, ages_series, genders_series], axis=1)

# Preprocess and clean data
under4s = []
for i in range(len(data)):
    if data['Ages'].iloc[i] <= 4:
        under4s.append(data.iloc[i])

under4s = pd.DataFrame(under4s)
under4s = under4s.sample(frac=0.3)
data = data[(data['Ages'] > 4) & (data['Ages'] < 80)]
data = pd.concat([data, under4s], ignore_index=True)
data = data[data['Genders'] != 3]
data = data.dropna(subset=['Genders'])
data = data.reset_index(drop=True)
data = data.dropna(subset=['Genders'])

# Prepare data for training
x = []
y_age = data['Ages']
y_gender = data['Genders']

for i in range(len(data)):
    data['Images'].iloc[i] = data['Images'].iloc[i].resize((224, 224), Image.BILINEAR)
    ar = np.asarray(data['Images'].iloc[i])
    x.append(ar)

x = np.array(x)

# Split data for age and gender prediction
x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(x, y_age, test_size=0.2, stratify=y_age)
x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(x, y_gender, test_size=0.2, stratify=y_gender)

# Define U-Net model for segmentation (you can modify the architecture as needed)
# def unet_model(input_shape):
#     inputs = Input(input_shape)
    
#     # Encoding
#     conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#     # Bottleneck
#     conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)

#     # Decoding
#     up1 = UpSampling2D(size=(2, 2))(conv3)
#     concat1 = Concatenate()([conv2, up1])

#     conv4 = Conv2D(128, 3, activation='relu', padding='same')(concat1)
#     conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
#     up2 = UpSampling2D(size=(2, 2))(conv4)
#     concat2 = Concatenate()([conv1, up2])

#     # Output
#     segmentation_output = Conv2D(1, 1, activation='sigmoid')(concat2)
def unet_model(input_size = (224,224,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2DTranspose(512,2,strides=(2,2),padding='same')(drop5)
    merge6 = concatenate([drop4,up6], axis = 1)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(256,2,strides=(2,2),padding='same')(conv6)
    merge7 = concatenate([conv3,up7], axis = 1)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(128,2,strides=(2,2),padding='same')(conv7)
    merge8 = concatenate([conv2,up8], axis = 1)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2DTranspose(64,2,strides=(2,2),padding='same')(conv8)
    merge9 = concatenate([conv1,up9], axis = 1)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'softmax')(conv9)

    # model = Model(inputs, conv10)

    # age_output = Conv2D(6, 1, activation='softmax', name='age_output')(model)

    # Output for gender
 #   gender_output = Conv2D(2, 1, activation='sigmoid', name='gender_output')(model)

    model = Model(inputs=inputs, outputs=[conv10], name='age_output')

    return model

# Create U-Net model
input_shape = (224, 224, 3)
segmentation_model = unet_model(input_shape)

# Compile U-Net model
segmentation_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load MobileNetV2 model for age prediction
age_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze layers in MobileNetV2
for layer in age_model.layers:
    layer.trainable = False

# Create Age Prediction Head
x = age_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
age_predictions = Dense(1, activation='linear')(x)  # Use activation='linear' for age prediction
age_model = Model(inputs=age_model.input, outputs=age_predictions)

# Compile Age Prediction Model
age_model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

# Concatenate U-Net output and age prediction output
combined_output = Concatenate()([segmentation_model.output, age_model.output])

# Create the combined model
combined_model = Model(inputs=[segmentation_model.input, age_model.input], outputs=combined_output)

# Compile the combined model
combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation for age prediction
age_datagen = ImageDataGenerator(
    rescale=1./255.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Prepare train and test data generators for age prediction
train_age = age_datagen.flow(x_train_age, y_train_age, batch_size=8)
test_age = age_datagen.flow(x_test_age, y_test_age, batch_size=8)

# Train the age prediction model
history_age = age_model.fit(train_age, epochs=5, shuffle=True, validation_data=test_age)
age_model.save('model_age_beta.h5')
