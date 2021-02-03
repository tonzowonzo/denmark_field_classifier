import rasterio
import numpy as np
import os
from tensorflow import one_hot
import tensorflow as tf
img_size = 256
classes = 302


def generator():
    # Open X data.
    for iiteration in range(3):
        for image_path in os.listdir(f"C:/Users/tim.iles/Documents/agriculture_data/s2_denmark/"):

            with rasterio.open(f"C:/Users/tim.iles/Documents/agriculture_data/s2_denmark/{image_path}") as src1:
                img = src1.read()
                img = np.moveaxis(img, 0, 2)

            print(img.shape)
            # Open labels data.
            with rasterio.open("C:/Users/tim.iles/Documents/agriculture_data/fields/clipped_field.tif") as src2:
                labels = src2.read()
                labels = labels.reshape(labels.shape[1:])

            no_data = labels.max() + 1
            labels[labels == -1] = no_data
            labels[labels < no_data] = 0
            labels[labels == no_data] = 1
            print(labels.max())
            print(labels.min())
            print(labels.shape)
            val = 0
            for i in range(0, 10700, img_size):
                for j in range(0, 10700, img_size):
                    X = img[i:i+img_size, j:j+img_size, :-1]
                    y = labels[i:i+img_size, j:j+img_size]
                    # y = one_hot(y, classes)
                    # y = y.numpy()
                    y = y.reshape((1, img_size, img_size, 1))
                    X = X.reshape(1, img_size, img_size, 10)
                    X = X / 10000
                    val += 1

                    yield X, y


if __name__ == '__main__':
    gen = generator()
    for i in range(100000):
        X, y = next(gen)
