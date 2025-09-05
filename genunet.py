import pandas as pd
import numpy as np
import seaborn as sns
import os
import PIL
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import UpSampling2D,Concatenate
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
sns.histplot(data['Ages'], kde=True, bins=30)
# plt.show()
data = data[data['Genders'] != 3]
# sns.countplot(data['Genders'])
sns.countplot(data=data, x='Genders')
plt.show()
x = []
y = []

#chuẩn hóa dữ liệu để huấn luyện
for i in range(len(data)):
    data['Images'].iloc[i] = data['Images'].iloc[i].resize((200, 200), Image.BILINEAR)

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
    
    # Mã hóa
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Giải mã
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    up1 = UpSampling2D(size=(2, 2))(conv2)
    concat1 = Concatenate()([conv1, up1])

    # Cuối cùng, dự đoán giới tính
    gender_output = Conv2D(1, 1, activation='linear')(concat1)

    model = Model(inputs=inputs, outputs=gender_output)
    return model

# Tạo mô hình U-Net dự đoán giới tính
input_shape = (200, 200, 3)  # Điều chỉnh kích thước ảnh đầu vào tùy theo dữ liệu của bạn
gendermodel = gender_unet_model(input_shape)

# Biên dịch mô hình với hàm mất mát và trình tối ưu
gendermodel.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

gender_datagen = ImageDataGenerator(
    rescale=1./255.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
test_gender = ImageDataGenerator(rescale=1./255)
train_gender = gender_datagen.flow(x_train_age, y_train_age, batch_size=8)
test_gender = gender_datagen.flow(x_test_age, y_test_age, batch_size=8)
history_gender = gendermodel.fit(train_gender, epochs=10, shuffle=True, validation_data=test_gender)
gendermodel.save('age_beta.h5')
