import numpy as np
from settings import PATH_DATA
from utils import center_image, create_xy, ORIGINAL_SHAPE
from tensorflow.keras.preprocessing.image import load_img

x_arrays, y_arrays = [], []
for image_path in PATH_DATA.data.images.iterdir():
    img = load_img(image_path, target_size=ORIGINAL_SHAPE)
    img_array = center_image(img)
    x_arr, y_arr = create_xy(img_array)
    x_arrays.append(x_arr)
    y_arrays.append(y_arr)

train_size = int(len(x_arrays) * 0.8)

x_arrays = np.array(x_arrays) / 255
y_arrays = np.array(y_arrays) / 255
x_arrays = x_arrays.astype(np.float32)
y_arrays = y_arrays.astype(np.float32)

for split in ['x', 'y']:
    np.save(PATH_DATA.data.arrays.joinpath(f'{split}_train.npy'), globals()[f'{split}_arrays'][:train_size])
    np.save(PATH_DATA.data.arrays.joinpath(f'{split}_test.npy'), globals()[f'{split}_arrays'][train_size:])
