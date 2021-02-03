from data_loader import generator
from keras.models import load_model
import cv2
import numpy as np
unet = load_model("C:/Users/tim.iles/Documents/unet_classifier.h5")

gen = generator()

for i in range(20):
    X, y = next(gen)
    pred = unet.predict(X)
    pred = pred.reshape((256, 256))
    X = X.reshape((256, 256, 10))
    X = (X[:, :, :3] * 10000).astype(np.uint16)
    cv2.imwrite(f"C:/Users/tim.iles/Documents/denmark_classif_{i}.tif", pred)
    cv2.imwrite(f"C:/Users/tim.iles/Documents/denmark_classif_X_{i}.tif", X)