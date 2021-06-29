import numpy as np
import json
import requests
import datetime as dt
import random

def Get_X(d1,m1,y1):
    date1 = dt.datetime(y1, m1, d1)
    date2 = date1 - dt.timedelta(days=6)
    d2 = date2.day
    m2 = date2.month
    y2 = date2.year

    headers = {'Authorization': 'ApiToken f058958a-d8bd-47cc-95d7-7ecf98610e47'}

    #Tempatures
    TDurl = "https://api.ims.gov.il/v1/envista/stations/178/data/7?from="+str(y2)+"/"+str(m2)+"/"+str(d2)+"&to="+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", TDurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    TD = np.array([i["channels"][0]["value"] for i in req_result["data"]])
#print(TD)

    #Rain %
    Rainurl = "https://api.ims.gov.il/v1/envista/stations/178/data/1?from="+str(y2)+"/"+str(m2)+"/"+str(d2)+"&to="+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", Rainurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    Rain = np.array([i["channels"][0]["value"] for i in req_result["data"]])
    for i in range(0,len(Rain)):
        if Rain[i] == 0:
            Rain[i] = 1
#(Rain)

    #Wind speed
    WSurl = "https://api.ims.gov.il/v1/envista/stations/178/data/4?from="+str(y2)+"/"+str(m2)+"/"+str(d2)+"&to="+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", WSurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    WS = np.array([i["channels"][0]["value"] for i in req_result["data"]])
    for i in range(0,len(WS)):
        if WS[i] == 0:
            WS[i] = 0.1
#print(WS)

    #Wind direction
    WDurl = "https://api.ims.gov.il/v1/envista/stations/178/data/5?from="+str(y2)+"/"+str(m2)+"/"+str(d2)+"&to="+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", WDurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    WD = np.array([i["channels"][0]["value"] for i in req_result["data"]])
    for i in range(0,len(WD)):
        if WD[i] == 0:
            WD[i] = 365
#print(WD)

    #Humidity
    RHurl = "https://api.ims.gov.il/v1/envista/stations/178/data/8?from="+str(y2)+"/"+str(m2)+"/"+str(d2)+"&to="+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", RHurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    RH = np.array([i["channels"][0]["value"] for i in req_result["data"]])
#print(RH)

    #Time
    Timeurl = "https://api.ims.gov.il/v1/envista/stations/178/data/13?from="+str(y2)+"/"+str(m2)+"/"+str(d2)+"&to="+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", Timeurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    Time = np.array([i["channels"][0]["value"] for i in req_result["data"]])
    for i in range(0,len(Time)):
         if Time[i] == 0:
             Time[i] = 24
#print(Time)

    data = np.array([TD, Rain, RH, WS, WD, Time]).T
    X = np.resize(data, (data.shape[0] * data.shape[1]))
    return X

def Get_Y(d1,m1,y1):
    headers = {'Authorization': 'ApiToken f058958a-d8bd-47cc-95d7-7ecf98610e47'}

    # Tempatures
    TDurl = "https://api.ims.gov.il/v1/envista/stations/178/data/7/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", TDurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    TD = np.array([i["channels"][0]["value"] for i in req_result["data"]])
    L = len(TD)
    TDAvg = 0
    for i in range(L):
        TDAvg += TD[i] / L
    # print(TD)

    # Rain %
    Rainurl = "https://api.ims.gov.il/v1/envista/stations/178/data/1/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", Rainurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    Rain = np.array([i["channels"][0]["value"] for i in req_result["data"]])
    RainAvg = 0
    for i in range(L):
        RainAvg += Rain[i] / L
    # (Rain)

    # Wind speed
    WSurl = "https://api.ims.gov.il/v1/envista/stations/178/data/4/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", WSurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    WS = np.array([i["channels"][0]["value"] for i in req_result["data"]])
    WSAvg = 0
    for i in range(L):
        WSAvg += WS[i] / L
    # print(WS)

    # Wind direction
    WDurl = "https://api.ims.gov.il/v1/envista/stations/178/data/5/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", WDurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    WD = np.array([i["channels"][0]["value"] for i in req_result["data"]])
    WDAvg = 0
    for i in range(L):
        WDAvg += WD[i] / L
    # print(WD)

    # Humidity
    RHurl = "https://api.ims.gov.il/v1/envista/stations/178/data/8/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", RHurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    RH = np.array([i["channels"][0]["value"] for i in req_result["data"]])
    RHAvg = 0
    for i in range(L):
        RHAvg += RH[i] / L
    # print(RH)

    # Time
    Timeurl = "https://api.ims.gov.il/v1/envista/stations/178/data/13/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", Timeurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    Time = np.array([i["channels"][0]["value"] for i in req_result["data"]])
    TimeAvg = 0
    for i in range (L):
        TimeAvg += Time[i]/L
    # print(Time)

    Y = []
    # Y = [storm, rainy ,strong winds, very cold, cold, very hot, hot, nice]
    if RainAvg >= 30:
        if WSAvg >= 7.5:  # storm
            Y = [1, 0, 0, 0, 0, 0, 0, 0]
        else:  # rainy
            Y = [0, 1, 0, 0, 0, 0, 0, 0]
    elif TDAvg < 22:
        if WSAvg >= 7.5:  # strong Winds
            Y = [0, 0, 1, 0, 0, 0, 0, 0]
        elif TDAvg <= 15:  # very cold
            Y = [0, 0, 0, 1, 0, 0, 0, 0]
        else:  # just cold
            Y = [0, 0, 0, 0, 1, 0, 0, 0]
    elif TDAvg >= 22:
        if TDAvg >= 30:
            if RH >= 70:  # very hot
                Y = [0, 0, 0, 0, 0, 1, 0, 0]
            else:  # just hot
                Y = [0, 0, 0, 0, 0, 0, 1, 0]
        else:  # nice
            Y = [0, 0, 0, 0, 0, 0, 0, 1]
    Y = np.transpose(Y)
    return Y

def NextDay(d1,m1,y1):

    date1 = dt.datetime(y1,m1,d1)
    date2 = date1 + dt.timedelta(days=1)
    d2 = date2.day
    m2 = date2.month
    # print(d2)
    # print(m2)
    y2 = date2.year
    return d2,m2,y2

def Isbigger(d1,m1,y1,d2,m2,y2):
    date1 = dt.date(y1,m1,d1)
    date2 = dt.date(y2,m2,d2)
    if date1>date2:
        return False
    return True

def Get_Data_from_to(d1,m1,y1,d2,m2,y2):

    days=0
    labels_train =[]
    labels_test = []
    x_train = []
    x_test = []
    train_test_ratio = 0.2
    while(Isbigger(d1,m1,y1,d2,m2,y2)):
        print(days)
        # print("d1 ="+str(d1)+"\nm1 ="+str(m1)+"\ny1 ="+str(y1))
        data=Get_X(d1,m1,y1)
        d3, m3, y3 = NextDay(d1, m1, y1)
        if (data.shape[0]!=5184):
            d1, m1, y1 = NextDay(d1, m1, y1)
            continue
        if (days==0):
            x_train.append(data)
            labels_train.append(Get_Y(d3,m3,y3))
        elif days==1:
            x_test.append(data)
            labels_test.append(Get_Y(d3,m3,y3))
        else:
            ratio=random.uniform(0, 1)
            if (ratio>train_test_ratio):
                x_train.append(data)
                labels_train.append(Get_Y(d3,m3,y3))
            else:
                x_test.append(data)
                labels_test.append(Get_Y(d3,m3,y3))

        # data=Get_X(d1,m1,y1)
        d1,m1,y1 = NextDay(d1,m1,y1)
        days+=1
    print("days="+str(days))
    x_train = np.array(x_train).T
    labels_train = np.array(labels_train).T
    x_test = np.array(x_test).T
    labels_test = np.array(labels_test).T
    return x_train,labels_train,x_test,labels_test.T

def GetSeason(m1):
    if m1>4 & m1 <10:
        return "summer"
    return "winter"
