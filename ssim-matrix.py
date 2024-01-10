import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Load two example images
l=0
for i in range(0,10):

    image1 = cv2.imread(f'result/ssim/gt/{i}.png', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(f'result/ssim/my/{i}.png', cv2.IMREAD_GRAYSCALE)

    # Ensure the images have the same shape
    min_height = min(image1.shape[0], image2.shape[0])
    min_width = min(image1.shape[1], image2.shape[1])
    image1 = image1[:min_height, :min_width]
    image2 = image2[:min_height, :min_width]

    # Compute SSIM value for the entire images
    ssim_value, _ = ssim(image1, image2, full=True)

    print(f"SSIM Value: {ssim_value}")
    l+=ssim_value
    # Display the two images
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    plt.title('Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    plt.title('Image 2')

    # plt.show()

print(l/10)
