import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_edges(image, method):
    if method == 'gradient_1':
        edges = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    elif method == 'canny':
        edges = cv2.Canny(image, 50, 150)
    else:
        raise ValueError("Invalid method. Supported methods: gradient_1, canny")

    return edges

def main():
    # Load grayscale image
    image = cv2.imread('Einstein.jpg', cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded successfully
    if image is None:
        print("Error: Could not read the image.")
        return

    # Apply edge detection using three different methods
    edges_gradient_1 = detect_edges(image, 'gradient_1')

    edges_canny = detect_edges(image, 'canny')

    # Display the original image and the detected edges using three methods
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(132)
    plt.imshow(np.abs(edges_gradient_1), cmap='gray')
    plt.title('Edges - Gradient 1')

    plt.subplot(133)
    plt.imshow(np.abs(edges_canny), cmap='gray')
    plt.title('Edges - Canny')

    plt.show()

if __name__ == "__main__":
    main()
