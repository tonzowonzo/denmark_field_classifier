from model import unet
from data_loader import generator
gen = generator()
unet_model = unet()
unet_model.fit(gen, steps_per_epoch=2500, epochs=3)

unet_model.save("C:/Users/tim.iles/Documents/unet_classifier.h5")