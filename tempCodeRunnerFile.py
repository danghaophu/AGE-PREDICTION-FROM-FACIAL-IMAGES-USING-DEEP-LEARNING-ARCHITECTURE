import pandas as pd
import numpy as np
import seaborn as sns
import os
import PIL
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import UpSampling2D,Concatenate
from keras.utils import to_categorical
#tạo path đến thư mục chứa ảnh 
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
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


# Tải lại mô hình
loaded_model = load_model('gender_age_beta.h5')

# Sử dụng mô hình để dự đoán trên dữ liệu kiểm tra
y_pred = loaded_model.predict(x_test_gender)

# Chuyển đổi dự đoán thành nhãn nhị phân (0 hoặc 1) dựa trên ngưỡng 0.5
y_pred_binary = (y_pred > 0.5).astype(int)

# Tính toán accuracy bằng cách so sánh nhãn dự đoán (y_pred_binary) với nhãn thực tế (y_test_gender)
accuracy = accuracy_score(y_test_gender, y_pred_binary)
print(f'Accuracy: {accuracy}')