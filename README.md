# 🧑‍🤝‍🧑 Age & Gender Prediction using U-Net

## 📌 Giới thiệu
Dự án này được xây dựng dựa trên nghiên cứu:  
**[Age Prediction from Facial Images Using Deep Learning Architecture](https://doi.org/10.2478/acss-2024-0018)** – Applied Computer Systems, Vol. 29, Issue 2, 2024.  
Dự án này được làm cùng với nhóm của tôi trong đó tôi được phân công xây dựng mô hình Unet 
Mục tiêu: **Dự đoán tuổi và giới tính từ ảnh khuôn mặt** với các mô hình học sâu:
- U-Net  
- MobileNets  
- EfficientNets  

---

## 🔬 Vai trò của U-Net
- U-Net được sử dụng cho **dự đoán giới tính**.  
- Đạt **độ chính xác 97.22%** – cao nhất trong các mô hình được so sánh.  
- Kiến trúc **encoder–decoder với skip connections** giúp trích xuất đặc trưng ở nhiều cấp độ, tối ưu cho việc phân biệt giới tính từ đặc trưng khuôn mặt tinh vi.  
- Tuy nhiên do giới hạn phần cứng nhóm rất tiết đã không triển khai toàn diện thành một ứng dụng nào.
---
## 📊 Kết quả thực nghiệm

![Experimental Results](https://github.com/user-attachments/assets/4d311929-0d78-4683-8991-dcbafe8db68a)

---

📷 Kết quả quá trình huấn luyện mô hình **U-Net**:

![Training Results](https://github.com/user-attachments/assets/ca1e0b33-f2ec-4afa-964e-255417c53dce)

---
## ⚙️ Công nghệ
- Python  
- TensorFlow / PyTorch  
- CNN Architectures: U-Net, MobileNets, EfficientNets  
- Data preprocessing: Chuẩn hóa, resize, augmentation ảnh khuôn mặt  

---

## 🚀 Cách chạy

```bash
# Clone repo
git clone https://github.com/yourusername/your-repo.git
cd your-repo

# Cài đặt thư viện
pip install -r requirements.txt

# Train mô hình U-Net
python train_unet.py --dataset ./data

# Dự đoán từ ảnh
python predict.py --image ./sample_face.jpg
