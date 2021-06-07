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
X = np.array([Get_X(24,5,2021),Get_X(23,2,2021),Get_X(22,1,2021),Get_X(21,4,2021)]).transpose()
Y = np.array([Get_Y(25,5,2021),Get_Y(24,2,2021),Get_Y(23,1,2021),Get_Y(22,4,2021)]).transpose()
print(X.shape)
print(Y.shape)
model = DLModel("Weather Forcast")
model.add(DLLayer("#1", 50, (X.shape[0],),"trim_tanh", "He", 0.01, "adaptive"))
model.add(DLLayer("#2", 10, (50,), "trim_tanh", "He", 0.01, "adaptive"))
#model.add(DLLayer("#3", 15, (50,), "trim_tanh", "He", 0.01, "adaptive"))
#model.add(DLLayer("#4", 10, (15,), "trim_tanh", "He", 0.01, "adaptive"))
model.add(DLLayer("#3", 5, (10,), "trim_tanh", "He", 0.01, "adaptive"))
model.add(DLLayer("softmax", 8, (5,), "softmax", "He", 0.01, "adaptive"))
model.compile("categorical_cross_entropy")
model.train(X,Y,1000)
#model.predict(np.expand_dims(Get_Y(25,5,2021), axis=0).transpose())
#np.expand_dims()

