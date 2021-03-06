from WeatherDataBase import *
from DL4 import *
import matplotlib.pyplot as plt
import numpy as np

import tkinter
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from tkinter.messagebox import *

def EnterDate():
    print("Please enter the day:")
    d1 = int(input())
    print("Please enter the month:")
    m1 = int(input())
    print("Please enter the year:")
    y1 = int(input())
    return d1, m1, y1


# model.train(X,Y,500)
# model.predict(np.expand_dims(Get_X(26,1,2021), axis=0).transpose())

def center_window(w=300, h=200):
    # get screen width and height
    ws = window.winfo_screenwidth()
    hs = window.winfo_screenheight()
    # calculate position x, y
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    window.geometry('%dx%d+%d+%d' % (w, h, x, y))

def setTrain():
    global train_test
    train_test="train"

def setTest():
    global train_test
    train_test="test"

window = Tk()
window.geometry("680x350")
center_window(680, 350)
window.title("Train or Test")
Label(window, text ="Please choose train/test...", font=('times', 24)).pack()
btn1=tkinter.Button(window, width=15, height=2, text="Training",font=('times', 24),command = lambda:[setTrain(),window.destroy()]).place(x=50,y=100)
btn2=tkinter.Button(window, width=15, height=2, text="Testing",font=('times', 24),command =lambda:[setTest(),window.destroy()]).place(x=350,y=100)
label1=Label(window, text ="Testing will take pretrained parameters...",anchor = 'center', font=('times', 24)).place(x=50,y=250)
window.mainloop()

cur_path = r'C:\Users\Itai Scherf\PycharmProjects\pythonProject1\paramaters summer'
# train_path = os.path.join(cur_path, 'data\\train')
train_list = ["storm awaits!", "rainy", "strong winds", "very cold", "cold", "very hot", "hot", "lovely day"]
# trainDict = {i: train_list[i] for i in range(0, len(train_list))}
# test_path = os.path.join(cur_path, 'data\\test')

# X_train, Y_train, X_test, Y_test = Get_Data_from_to(1, 5, 2021, 29, 6, 2021)
with h5py.File(r'C:\Users\Itai Scherf\PycharmProjects\pythonProject1\Database\WeatherData.h5', 'r') as hf:
    X_train = hf['x_train'][:]
    Y_train = hf['labels_train'][:]
    X_test = hf['x_test'][:]
    Y_test = hf['labels_test'][:]

if (train_test == 'train'):


    print("X train shape:" + str(X_train.shape))
    print("X test shape:" + str(X_test.shape))
    print("Y train shape:" + str(Y_train.shape))
    print("Y test shape:" + str(Y_test.shape))

    W_init1 = "Xaviar"
    W_init2 = "Xaviar"
    W_init3 = "Xaviar"
    W_init4 = "Xaviar"
    W_init5 = "Xaviar"
    W_init6 = "Xaviar"
    layer1 = DLLayer("#1", 100, (X_train.shape[0],), "trim_tanh", W_init1, 0.01, "adaptive")
    layer2 = DLLayer("#2", 50, (100,), "trim_tanh", W_init2, 0.01, "adaptive")
    layer3 = DLLayer("#3", 50, (50,), "trim_tanh", W_init3, 0.01, "adaptive")
    layer4 = DLLayer("#4", 50, (50,), "trim_tanh", W_init4, 0.01, "adaptive")
    layer5 = DLLayer("#5", 20, (50,), "trim_tanh", W_init5, 0.01, "adaptive")
    layer6 = DLLayer("softmax", 8, (20,), "trim_softmax", "Xaviar", 0.01, "adaptive")

    model = DLModel("Weather Forcast")
    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
    model.add(layer4)
    model.add(layer5)
    model.add(layer6)
    model.compile("categorical_cross_entropy")

    cost = model.train(X_train, Y_train, 500)
    plt.plot(np.squeeze(cost))
    plt.ylabel("cost")
    plt.xlabel("iterations")
    plt.title("Learning rate = 0.01")
    plt.show()

    print("Train:...")
    # for logistic/binary regretion:
    print("Train accuracy:", np.mean(model.predict(X_train) == Y_train))
    # for softmax:
    model.confusion_matrix(X_train, Y_train)
    print()
    print("Test:...")
    # for logistic/binary regretion:
    print("Test accuracy:", np.mean(model.predict(X_test) == Y_test))
    # for softmax:
    model.confusion_matrix(X_test, Y_test)
    print()

    print("Saving weights...")
    model.save_weights(r'C:\Users\Itai Scherf\PycharmProjects\pythonProject1\paramaters summer')
##
    d1, m1, y1 = EnterDate()
    print("please wait a moment ????")
    X = np.expand_dims(Get_X(d1, m1, y1), axis=0).transpose()

    layer1 = DLLayer("#1", 100, (X.shape[0],), "trim_tanh", W_init1, 0.01, "adaptive")
    layer2 = DLLayer("#2", 50, (100,), "trim_tanh", W_init2, 0.01, "adaptive")
    layer3 = DLLayer("#3", 50, (50,), "trim_tanh", W_init3, 0.01, "adaptive")
    layer4 = DLLayer("#4", 50, (50,), "trim_tanh", W_init4, 0.01, "adaptive")
    layer5 = DLLayer("#5", 20, (50,), "trim_tanh", W_init5, 0.01, "adaptive")
    layer6 = DLLayer("softmax", 8, (20,), "trim_softmax", "Xaviar", 0.01, "adaptive")

    model = DLModel("Weather Forcast")
    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
    model.add(layer4)
    model.add(layer5)
    model.add(layer6)
    model.compile("categorical_cross_entropy")

    predictions = model.predict(X)
    print(predictions.transpose())
else:

    W_init1 = cur_path + "\Layer1.h5"
    W_init2 = cur_path + "\Layer2.h5"
    W_init3 = cur_path + "\Layer3.h5"
    W_init4 = cur_path + "\Layer4.h5"
    W_init5 = cur_path + "\Layer5.h5"

    d1, m1, y1 = EnterDate()
    print("please wait a moment ????")
    X = np.expand_dims(Get_X(d1, m1, y1), axis=0).transpose()

    layer1 = DLLayer("#1", 100, (X.shape[0],), "trim_tanh", W_init1, 0.01, "adaptive")
    layer2 = DLLayer("#2", 50, (100,), "trim_tanh", W_init2, 0.01, "adaptive")
    layer3 = DLLayer("#3", 50, (50,), "trim_tanh", W_init3, 0.01, "adaptive")
    layer4 = DLLayer("#4", 50, (50,), "trim_tanh", W_init4, 0.01, "adaptive")
    layer5 = DLLayer("#5", 20, (50,), "trim_tanh", W_init5, 0.01, "adaptive")
    layer6 = DLLayer("softmax", 8, (20,), "trim_softmax", "Xaviar", 0.01, "adaptive")

    model = DLModel("Weather Forcast")
    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
    model.add(layer4)
    model.add(layer5)
    model.add(layer6)
    model.compile("categorical_cross_entropy")

    predictions = model.predict(X)
    # ans = trainDict[DLModel.conv(predictions)]
    # root = Tk()
    # root.overrideredirect(1)
    # root.withdraw()
    # messagebox.showinfo(ans)
    # root.destroy()
