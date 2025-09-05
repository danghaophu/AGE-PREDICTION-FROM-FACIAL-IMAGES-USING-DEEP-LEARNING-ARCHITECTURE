import pandas as pd
import numpy as np
import seaborn as sns
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from keras.layers import  Dense
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import  Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from segmentation_models import Unet
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
    data['Images'] = data['Images'].apply(lambda x: x.resize((256, 256), Image.BILINEAR))
    ar = data['Images'].iloc[i]
    x.append(ar)
    agegen = [int(data['Ages'].iloc[i]), int(data['Genders'].iloc[i])]
    y.append(agegen)
x = x
y_gender = data['Genders']

x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(x, y_gender, test_size=0.2, stratify=y_gender)

model= Unet('resnet34',encoder_weights="imagenet",input_shape=(256,256,3),encoder_freeze=True)
model.trainable = True

y = model.output
#x = BatchNormalization() (x)
y = Dense(128, activation = "relu") (y)
#x = Dropout(0.3) (x)
output2 = Dense(1, activation = "sigmoid") (y)
# Biên dịch mô hình với hàm mất mát và trình tối ưu
gendermodel = Model(inputs = model.input, outputs = output2)
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
history_gender = gendermodel.fit(train_gender, epochs=2, shuffle=True, validation_data=test_gender)
gendermodel.save('gender_beta_unet.h5')
