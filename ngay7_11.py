import numpy as np
import cv2
import matplotlib.pyplot as plt

# Bước 1: Đọc ảnh
image = cv2.imread('dv.png')  # Thay đổi đường dẫn đến ảnh của bạn
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB
pixel_values = image.reshape((-1, 3))  # Chuyển đổi ảnh thành mảng 2D
pixel_values = np.float32(pixel_values)  # Đổi kiểu dữ liệu thành float

# Bước 2: Xác định các giá trị k
k_values = [2, 3, 4, 5]

# Bước 3: Tạo một figure để hiển thị nhiều subplot
plt.figure(figsize=(15, 10))

# Bước 4: Thực hiện K-means cho từng giá trị k
for index, k in enumerate(k_values):
    # Khởi tạo các centroid ngẫu nhiên
    np.random.seed(42)  # Để có kết quả tái lập
    centroids = pixel_values[np.random.choice(pixel_values.shape[0], k, replace=False)]# Chọn ngẫu nhiên k điểm dữ liệu làm centroid

    # Bước 5: Lặp cho đến khi hội tụ
    for i in range(100):  # Giới hạn số lần lặp
        # Bước 6: Gán các điểm dữ liệu cho các centroid
        distances = np.zeros((pixel_values.shape[0], k))  # Tạo mảng khoảng cách
        for j in range(k):
            distances[:, j] = np.linalg.norm(pixel_values - centroids[j], axis=1)# Tính khoảng cách từ mỗi điểm dữ liệu đến các centroid

        labels = np.argmin(distances, axis=1)  # Gán nhãn cho các điểm

        # Bước 7: Cập nhật các centroid
        new_centroids = np.zeros((k, 3))  # Tạo mảng mới cho centroid
        for j in range(k):
            if np.any(labels == j):  # Kiểm tra có điểm nào thuộc cụm không
                new_centroids[j] = pixel_values[labels == j].mean(axis=0)

        # Kiểm tra hội tụ
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    # Bước 8: Tạo hình ảnh phân cụm
    segmented_image = np.zeros_like(pixel_values)  # Tạo mảng cho hình ảnh phân cụm
    for j in range(k):# Gán mỗi điểm dữ liệu với centroid g
        segmented_image[labels == j] = centroids[j]

    segmented_image = segmented_image.reshape(image.shape)

    # Bước 9: Hiển thị kết quả trong subplot
    plt.subplot(2, 2, index + 1)  # Sắp xếp trong 2 hàng, 2 cột
    plt.imshow(segmented_image.astype(np.uint8))
    plt.title(f'K-means Segmentation with K = {k}')
    plt.axis('off')
# Bước 10: Hiển thị tất cả các hình ảnh
plt.tight_layout()
plt.show()
