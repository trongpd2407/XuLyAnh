import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

# Hàm xử lý ảnh
def process_image(process_function):
    global img_original
    img_processed = process_function(img_original)
    display_image(img_processed)

# Hàm chọn ảnh từ file
def open_image():
    global img_original
    file_path = filedialog.askopenfilename()
    img_original = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    display_image(img_original)

# Hàm hiển thị ảnh trên giao diện
def display_image(img):
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img

# Hàm xử lý âm bản
def apply_negative(img):
    return 255 - img

# Hàm xử lý phân ngưỡng
def apply_threshold(img):
    _, thresholded = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return thresholded

# Hàm xử lý biến đổi logarit
def apply_log_transform(img):
    c = 255 / (np.log(1 + img.max()))
    log_transformed = c * (np.log(1 + img))
    log_transformed = log_transformed.astype('uint8')
    return log_transformed

# Hàm xử lý biến đổi hàm mũ
def apply_power_transform(img, gamma=1.5):
    power_transformed = np.power(img, gamma)
    power_transformed = ((power_transformed / power_transformed.max()) * 255).astype('uint8')
    return power_transformed

# Hàm xử lý tăng độ tương phản
def apply_contrast_increase(img, alpha=1.5, beta=20):
    contrast_increased = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return contrast_increased

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Xử lý ảnh")

# Tạo nút và gắn kết với hàm xử lý
btn_open = tk.Button(root, text="Chọn ảnh", command=open_image)
btn_open.pack()

btn_negative = tk.Button(root, text="Âm bản", command=lambda: process_image(apply_negative))
btn_negative.pack()

btn_threshold = tk.Button(root, text="Phân ngưỡng", command=lambda: process_image(apply_threshold))
btn_threshold.pack()

btn_log_transform = tk.Button(root, text="Biến đổi logarit", command=lambda: process_image(apply_log_transform))
btn_log_transform.pack()

btn_power_transform = tk.Button(root, text="Biến đổi hàm mũ", command=lambda: process_image(apply_power_transform))
btn_power_transform.pack()

btn_contrast_increase = tk.Button(root, text="Tăng độ tương phản", command=lambda: process_image(apply_contrast_increase))
btn_contrast_increase.pack()

# Panel hiển thị ảnh
panel = tk.Label(root)
panel.pack()

# Chạy cửa sổ
root.mainloop()
