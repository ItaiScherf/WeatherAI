from WeatherDataBase import *
from DL4 import *
import matplotlib.pyplot as plt
import numpy as np

def EnterDate():
    print("Please enter the day:")
    d1 = input()
    print("Please enter the month:")
    m1 = input()
    print("Please enter the year:")
    y1 = input()
    return d1,m1,y1
# X = np.expand_dims(Get_X(24,5,2021), axis=0).transpose()
# Y = np.expand_dims(Get_Y(25,5,2021), axis=0).transpose()
# X = np.array([Get_X(24,1,2021),Get_X(23,1,2021),Get_X(22,1,2021),Get_X(21,1,2021),Get_X(20,1,2021),Get_X(19,1,2021),Get_X(18,1,2021),Get_X(17,1,2021),Get_X(16,1,2021),Get_X(15,1,2021),Get_X(14,1,2021),Get_X(13,1,2021),Get_X(12,1,2021),Get_X(11,1,2021),Get_X(10,1,2021),Get_X(9,1,2021),Get_X(8,1,2021)]).transpose()
# Y = np.array([Get_Y(25,1,2021),Get_Y(24,1,2021),Get_Y(23,1,2021),Get_Y(22,1,2021),Get_Y(21,1,2021),Get_Y(20,1,2021),Get_Y(19,1,2021),Get_Y(18,1,2021),Get_Y(17,1,2021),Get_Y(16,1,2021),Get_Y(15,1,2021),Get_Y(14,1,2021),Get_Y(13,1,2021),Get_Y(12,1,2021),Get_Y(11,1,2021),Get_Y(10,1,2021),Get_Y(9,1,2021)]).transpose()
# print(X.shape)
# print(Y.shape)

# model.train(X,Y,500)
# model.predict(np.expand_dims(Get_X(26,1,2021), axis=0).transpose())
# model.confusion_matrix(Get_X(26,1,2021),Get_Y(27,1,2021))
# np.expand_dims()


cur_path = "D:\אולטרה תיקייה\הנדסת תוכנה"
train_path = os.path.join(cur_path, 'data\\train')
train_list = ["storm awaits!", "rainy", "strong winds", "very cold", "cold", "very hot", "hot", "lovely day"]
trainDict = {i: train_list[i] for i in range(0, len(train_list))}
test_path = os.path.join(cur_path,'data\\test')


train_test = 'train'

if (train_test == 'train'):

    X_train, Y_train, X_test, Y_test = Get_Data_from_to(8, 1, 2021, 8, 4, 2021)

    W_init1 = "Xaviar"
    W_init2 = "Xaviar"
    W_init3 = "Xaviar"
    W_init4 = "Xaviar"
    W_init5 = "Xaviar"

else:

    W_init1 = cur_path + "\\Layer1.h5"
    W_init2 = cur_path + "\\Layer2.h5"
    W_init3 = cur_path + "\\Layer3.h5"
    W_init4 = cur_path + "\\Layer4.h5"
    W_init5 = cur_path + "\\Layer4.h5"

model = DLModel("Weather Forcast")
model.add(DLLayer("#1", 50, (X_train.shape[0],),"trim_tanh", "Xaviar", 0.01, "adaptive"))
model.add(DLLayer("#2", 20, (50,), "trim_tanh", "Xaviar", 0.01, "adaptive"))
model.add(DLLayer("#3", 20, (20,), "trim_tanh", "Xaviar", 0.01, "adaptive"))
model.add(DLLayer("#4", 10, (20,), "trim_tanh", "Xaviar", 0.01, "adaptive"))
model.add(DLLayer("#5", 10, (10,), "trim_tanh", "Xaviar", 0.01, "adaptive"))
model.add(DLLayer("softmax", 8, (10,), "trim_softmax", "Xaviar", 0.01, "adaptive"))
model.compile("categorical_cross_entropy")
if (train_test == 'train'):
    cost = model.train(X_train, Y_train, 1000)
    plt.plot(np.squeeze(cost))
    plt.ylabel("cost")
    plt.xlabel("iterations")
    plt.title("Learning rate = 0.01")
    plt.show()

    print("Train:...")
    # for logistic/binary regretion:
    print("Train accuracy:",np.mean(model.predict(X_train)==Y_train))
    # for softmax:
    model.confusion_matrix(X_train, Y_train)
    print()
    print("Test:...")
    # for logistic/binary regretion:
    print("Test accuracy:",np.mean(model.predict(X_test)==Y_test))
    # for softmax:
    model.confusion_matrix(X_test, Y_test)
    print()

    print("Saving weights...")
    model.save_weights(cur_path)

# Check on single date:
# while (1):
#     d1, m1, y1 = EnterDate()
#     x_train = Get_X(d1, m1, y1) / 360
#     predictions = model.predict(X_train)
#     ans = trainDict[DLModel.conv(predictions)]
#     root = Tk()
#     root.overrideredirect(1)
#     root.withdraw()
#     messagebox.showinfo(ans)
#     root.destroy()
