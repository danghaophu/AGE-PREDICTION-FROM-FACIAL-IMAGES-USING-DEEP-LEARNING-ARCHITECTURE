import pandas as pd
import numpy as np
import seaborn as sns
import os
import PIL
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense,Conv2DTranspose
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import UpSampling2D,Concatenate
from keras.utils import to_categorical
#tạo path đến thư mục chứa ảnh 
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

#tạo các pd.series cho các mảng
images = pd.Series(list(images), name='Images')
ages_series = pd.Series(ages, name='Ages')
genders_series = pd.Series(genders, name='Genders')
data = pd.concat([images, ages_series, genders_series], axis=1)

under4s = []
for i in range(len(data)):
    if data['Ages'].iloc[i] <= 4:
        under4s.append(data.iloc[i])
under4s = pd.DataFrame(under4s)
under4s = under4s.sample(frac=0.3)
data = data[(data['Ages'] > 4) & (data['Ages'] < 80)]
data = pd.concat([data, under4s], ignore_index=True)
# sns.histplot(data['Ages'], kde=True, bins=30)
# # plt.show()
data = data[data['Genders'] != 3]
sns.countplot(data['Genders'])
# sns.countplot(data=data, x='Genders')
# #plt.show()
data = data.dropna(subset=['Genders'])
data = data.reset_index(drop=True)
data = data.dropna(subset=['Genders'])


x = []
y = []

#chuẩn hóa dữ liệu để huấn luyện
for i in range(len(data)):
    data['Images'].iloc[i] = data['Images'].iloc[i].resize((256, 256), Image.BILINEAR)

    ar = np.asarray(data['Images'].iloc[i])
    x.append(ar)
    agegen = [int(data['Ages'].iloc[i]), int(data['Genders'].iloc[i])]
    y.append(agegen)
x = np.array(x)
y_age = data['Ages']
y_gender = data['Genders']

x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(x, y_age, test_size=0.2, stratify=y_age)
x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(x, y_gender, test_size=0.2, stratify=y_gender)

def gender_unet_model(input_shape):
    inputs = Input(input_shape)
    
    # Encoding
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    drop3 = Dropout(0.5)(conv3)
# Decoding
    up4 = Conv2DTranspose(128,2,strides=(2,2),padding='same')(drop3)
    concat4 = Concatenate()([conv2, up4])
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(concat4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
    up5 = Conv2DTranspose(64,2,strides=(2,2),padding='same')(conv4)
    concat5 = Concatenate()([conv1, up5])
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(concat5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    # Đầu ra giống mô hình MobileNetV2
    x = GlobalAveragePooling2D()(conv5)
    x = Dense(1, activation='relu')(x)
    gender_output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=gender_output)
    return model


input_shape = (256, 256, 3)
gendermodel = gender_unet_model(input_shape)

# Biên dịch lại mô hình với hàm mất mát 'binary_crossentropy'
gendermodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

gender_datagen = ImageDataGenerator(
    rescale=1./255.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
test_gender = ImageDataGenerator(rescale=1./255)
train_gender = gender_datagen.flow(x_train_gender, y_train_gender, batch_size=8)
test_gender = gender_datagen.flow(x_test_gender, y_test_gender, batch_size=8)
history_gender = gendermodel.fit(train_gender, epochs=10, shuffle=True, validation_data=test_gender)
gendermodel.save('gen_beta.h5')
