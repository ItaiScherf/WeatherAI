from WeatherDataBase import *
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from unit10 import c1w4_utils as u10
from DL4 import *

# X = np.expand_dims(Get_X(24,5,2021), axis=0).transpose()
# Y = np.expand_dims(Get_Y(25,5,2021), axis=0).transpose()
X = np.array([Get_X(5, 6, 2021), Get_X(4, 6, 2021), Get_X(3, 6, 2021), Get_X(2, 6, 2021)]).transpose()
Y = np.array([Get_Y(6, 6, 2021), Get_Y(5, 6, 2021), Get_Y(4, 6, 2021), Get_Y(3, 6, 2021)]).transpose()
# print(X.shape)
# print(Y.shape)
model = DLModel("Weather Forcast")
model.add(DLLayer("#1", 50, (X.shape[0],), "trim_tanh", "He", 0.01, "adaptive"))
model.add(DLLayer("#2", 10, (50,), "trim_tanh", "He", 0.01, "adaptive"))
# model.add(DLLayer("#3", 15, (50,), "trim_tanh", "He", 0.01, "adaptive"))
# model.add(DLLayer("#4", 10, (15,), "trim_tanh", "He", 0.01, "adaptive"))
model.add(DLLayer("#3", 5, (10,), "trim_tanh", "He", 0.01, "adaptive"))
model.add(DLLayer("softmax", 8, (5,), "softmax", "He", 0.01, "adaptive"))
model.compile("categorical_cross_entropy")
model.train(X, Y, 100)
X = model.predict(np.expand_dims(Get_X(6,6,2021), axis=0).transpose())
print(X.transpose())
# np.expand_dims()
