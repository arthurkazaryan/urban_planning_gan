from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from sklearn.cluster import KMeans
from utils import UrbanPlanningGAN
from settings import MainPathData
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys


def predict_image(pic_path):
    image = load_img(pic_path, target_size=(256, 384, 3))
    image = img_to_array(image).astype('uint8')
    image_for_pred = image.astype('float32')
    image_for_pred = (image_for_pred - 127.5) / 127.5
    predict = np.array(UPGAN.gan_model.predict(np.expand_dims(image_for_pred, axis=0))[1][0])

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j, 0] < 20 and image[i, j, 1] < 20 and image[i, j, 2] < 20:
                image[i, j] = [255, 255, 255]

    cluster_num = 2
    X = predict.reshape(-1, 3)
    km = KMeans(n_clusters=cluster_num)
    km.fit(X)
    labels = km.labels_
    mask = labels.reshape(256, 384)
    mask_shape = mask.shape

    for i in range(mask_shape[0]):
        for j in range(mask_shape[1]):
            if mask[i, j] == 1:
                image[i, j] = [0, 0, 0]

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(f"predict_{pic_path.name}", bbox_inches='tight')
    plt.show()
    plt.close()


UPGAN = UrbanPlanningGAN()
UPGAN.gan_model.load_weights(MainPathData.data.base.joinpath('gan_model.h5'))
predict_image(Path(sys.argv[1]))
