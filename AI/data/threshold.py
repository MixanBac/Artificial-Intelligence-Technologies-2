import cv2
import numpy as np

def threshold_image(image_path, threshold_value):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    thresholded_image = np.zeros_like(image)

    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            if image[i, j] > threshold_value:
                thresholded_image[i, j] = 255

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("/usr/app/src/output.jpg", thresholded_image)

image_path = "/usr/app/src/t.jpg"
threshold_value = 127  

threshold_image(image_path, threshold_value)
print('Успех в обработке!')

import os

target_folder = 'C:/Users/User/AI'

os.makedirs(target_folder, exist_ok=True)

import shutil

source_path = '/usr/app/src/output.jpg'

target_path = 'C:/Users/User/AI/output.jpg'

shutil.copy(source_path, target_path)
print('Успех в загрузке!')
