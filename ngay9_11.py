import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from skfuzzy.cluster import cmeans

# Đọc ảnh vệ tinh
image_path = 'vetinh.png'  # Đường dẫn đến ảnh vệ tinh
image = cv2.imread(image_path)

# Kiểm tra nếu ảnh không được tải
if image is None:
    print("Không thể đọc ảnh từ đường dẫn đã cho.")
else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB

    # Giảm kích thước ảnh xuống 25% để giảm dung lượng bộ nhớ sử dụng
    scale_percent = 25
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Chuyển đổi ảnh thành dữ liệu 2D
    pixel_values = image_resized.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Số lượng cụm
    n_clusters = 3

    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans_labels = kmeans.fit_predict(pixel_values)

    # Fuzzy C-Means (FCM) với skfuzzy
    cntr, u, _, _, _, _, _ = cmeans(pixel_values.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)
    fcm_labels = np.argmax(u, axis=0)

    # Agglomerative Hierarchical Clustering
    ahc = AgglomerativeClustering(n_clusters=n_clusters)
    ahc_labels = ahc.fit_predict(pixel_values)

    # Hàm tính số lượng điểm trong mỗi cụm
    def print_cluster_sizes(labels, method_name):
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Số lượng điểm trong các cụm ({method_name}):")
        for cluster, count in zip(unique, counts):
            print(f"Cụm {cluster}: {count} điểm")

    # Hiển thị số lượng điểm trong mỗi cụm
    print_cluster_sizes(kmeans_labels, "K-means")
    print_cluster_sizes(fcm_labels, "Fuzzy C-Means")
    print_cluster_sizes(ahc_labels, "Agglomerative Hierarchical Clustering")

    # Hiển thị kết quả
    plt.figure(figsize=(15, 5))

    # Ảnh gốc
    plt.subplot(1, 4, 1)
    plt.imshow(image_resized)
    plt.title('Original Image')
    plt.axis('off')

    # Hiển thị nhãn của K-means
    plt.subplot(1, 4, 2)
    plt.imshow(kmeans_labels.reshape(image_resized.shape[:2]), cmap='tab20')
    plt.title('K-means Labels')
    plt.axis('off')

    # Hiển thị nhãn của Fuzzy C-Means
    plt.subplot(1, 4, 3)
    plt.imshow(fcm_labels.reshape(image_resized.shape[:2]), cmap='tab20')
    plt.title('Fuzzy C-Means Labels')
    plt.axis('off')

    # Hiển thị nhãn của AHC
    plt.subplot(1, 4, 4)
    plt.imshow(ahc_labels.reshape(image_resized.shape[:2]), cmap='tab20')
    plt.title('AHC Labels')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
