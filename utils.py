import numpy as np
from scipy import ndimage
from random import randint

ORIGINAL_SHAPE = (int(564*0.35), int(1269*0.35), 3)
TARGET_SHAPE = (256, 384, 3)


def center_image(image):
    
    image = np.array(image, dtype=np.uint8)[:, :, :3]
    col_sum = np.where(np.sum(image, axis=0) != np.sum(image, axis=0)[0])
    row_sum = np.where(np.sum(image, axis=1) != np.sum(image, axis=1)[0])
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]
    cropped_image = image[y1:y2, x1:x2]

    crop_shape = cropped_image.shape
    h1 = int((TARGET_SHAPE[0] - crop_shape[0]) / 2)
    horizontal1 = np.full((h1, crop_shape[1], 3), (255, 255, 255)).astype('uint8')
    img = np.concatenate((horizontal1, cropped_image), axis=0)

    img_shape = img.shape
    h2 = TARGET_SHAPE[0] - img_shape[0]
    horizontal2 = np.full((h2, crop_shape[1], 3), (255, 255, 255)).astype('uint8')
    img = np.concatenate((img, horizontal2), axis=0)

    img_shape = img.shape
    v1 = int((TARGET_SHAPE[1] - img_shape[1]) / 2)
    vertical1 = np.full((TARGET_SHAPE[0], v1, 3), (255, 255, 255)).astype('uint8')
    img = np.concatenate((vertical1, img), axis=1)

    img_shape = img.shape
    v2 = TARGET_SHAPE[1] - img_shape[1]
    vertical2 = np.full((TARGET_SHAPE[0], v2, 3), (255, 255, 255)).astype('uint8')
    img = np.concatenate((img, vertical2), axis=1)
    
    return img


def create_xy(image: np.ndarray):
        
    shape = image.shape
    x = image.copy()
    y = np.zeros((shape[0], shape[1], shape[2]), dtype=np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if x[i][j][0] < 60 and x[i][j][1] < 60 and x[i][j][2] < 60:
                x[i][j] = x[i-1][j]
                y[i][j] = np.array([255, 255, 255])
            elif x[i][j][0] > 240 and x[i][j][1] > 240 and x[i][j][2] > 240:
                x[i][j] = np.array([0, 0, 0])
    x = x.astype('uint8')
    
    probability = randint(0, 1)
    if probability == 1:
        angle = randint(-45, 45)
        x = ndimage.rotate(x, angle, reshape=False)
        y = ndimage.rotate(y, angle, reshape=False)
    
    return x, y
