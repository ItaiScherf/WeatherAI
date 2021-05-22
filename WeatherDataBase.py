import numpy as np
import json
import requests
import datetime as dt

def Get_X(d1,m1,y1):

    if d1-7<0:
        if m1 == 1:
            d2 = d1 - 7 + 31
            m2 = 12
            y2 = y1-1
        elif m1 == 3 or m1 == 5 or m1 == 7 or m1 == 8 or m1 == 10 or m1 == 12:
            d2 = d1-7+31
            m2 = m1 - 1
            y2 = y1
        elif m1 == 4 or m1 == 6 or m1 == 9 or m1 == 11:
            d2 = d1 - 7 + 30
            m2 = m1 - 1
            y2 = y1
        elif  m1 == 2:
            d2 = d1 - 7 + 28
            m2 = m1 - 1
            y2 = y1
    else:
        d2 = d1 - 7
        m2 = m1
        y2 = y1
    print(str(d2)+"-"+str(m2)+"-"+str(y2))
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
#(Rain)

    #Wind speed
    WSurl = "https://api.ims.gov.il/v1/envista/stations/178/data/4?from="+str(y2)+"/"+str(m2)+"/"+str(d2)+"&to="+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", WSurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    WS = np.array([i["channels"][0]["value"] for i in req_result["data"]])
#print(WS)

    #Wind direction
    WDurl = "https://api.ims.gov.il/v1/envista/stations/178/data/5?from="+str(y2)+"/"+str(m2)+"/"+str(d2)+"&to="+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", WDurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    WD = np.array([i["channels"][0]["value"] for i in req_result["data"]])
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
#print(Time)

    data = np.array([TD, Rain, RH, WS, WD, Time])
    x = np.transpose(data)
    return x

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
    #Y = [storm, rainy ,strong winds, very cold, cold, nightmare, super hot, nice] 
    if RainAvg > 30:
        if WSAvg > 7.5: #storm
            Y = [1,0,0,0,0,0,0,0]
        else: #rainy
            Y = [0,1,0,0,0,0,0,0]
    elif TDAvg < 22:
        if WSAvg > 7.5: # strong Winds
            Y = [0,0,1,0,0,0,0,0]
        elif TDAvg < 15: #very cold
            Y = [0,0,0,1,0,0,0,0]
        else: # just cold
            Y = [0,0,0,0,1,0,0,0]
    elif TDAvg >= 22:
        if TDAvg >= 30:
            if RH>=70:# nightmare
                Y = [0,0,0,0,0,1,0,0]
            else: #superhot
                Y = [0,0,0,0,0,0,1,0]
        else: #nice
            Y = [0,0,0,0,0,0,0,1]


    return Y
