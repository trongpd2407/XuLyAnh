import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_erosion(image, kernel):
    """
    Áp dụng phép co (erosion) lên ảnh grayscale.

    :param image: Ảnh grayscale đầu vào.
    :param kernel: Structuring element cho phép co.
    :return: Ảnh sau khi áp dụng phép co.
    """
    result = cv2.erode(image, kernel, iterations=1)
    return result

def apply_dilation(image, kernel):
    """
    Áp dụng phép dãn (dilation) lên ảnh grayscale.

    :param image: Ảnh grayscale đầu vào.
    :param kernel: Structuring element cho phép dãn.
    :return: Ảnh sau khi áp dụng phép dãn.
    """
    result = cv2.dilate(image, kernel, iterations=1)
    return result

def main():
    # Đọc ảnh grayscale từ file
    image = cv2.imread('images.png', cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có được đọc thành công không
    if image is None:
        print("Lỗi: Không thể đọc ảnh.")
        return

    # Định nghĩa structuring element (kernel) tùy chọn
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Áp dụng phép co và phép dãn sử dụng structuring element
    eroded_image = apply_erosion(image, kernel)
    dilated_image = apply_dilation(image, kernel)

    # Hiển thị ảnh gốc và ảnh sau khi áp dụng phép co và phép dãn
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(132)
    plt.imshow(eroded_image, cmap='gray')
    plt.title('Ảnh Sau Khi Áp Dụng Phép Co')

    plt.subplot(133)
    plt.imshow(dilated_image, cmap='gray')
    plt.title('Ảnh Sau Khi Áp Dụng Phép Dãn')

    plt.show()

if __name__ == "__main__":
    main()
