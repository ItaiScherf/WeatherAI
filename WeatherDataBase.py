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
    # print(TD)

    # Rain %
    Rainurl = "https://api.ims.gov.il/v1/envista/stations/178/data/1/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", Rainurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    Rain = np.array([i["channels"][0]["value"] for i in req_result["data"]])
    # (Rain)

    # Wind speed
    WSurl = "https://api.ims.gov.il/v1/envista/stations/178/data/4/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", WSurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    WS = np.array([i["channels"][0]["value"] for i in req_result["data"]])
    # print(WS)

    # Wind direction
    WDurl = "https://api.ims.gov.il/v1/envista/stations/178/data/5/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", WDurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    WD = np.array([i["channels"][0]["value"] for i in req_result["data"]])
    # print(WD)

    # Humidity
    RHurl = "https://api.ims.gov.il/v1/envista/stations/178/data/8/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", RHurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    RH = np.array([i["channels"][0]["value"] for i in req_result["data"]])
    # print(RH)

    # Time
    Timeurl = "https://api.ims.gov.il/v1/envista/stations/178/data/13/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
    response = requests.request("GET", Timeurl, headers=headers)
    req_result = json.loads(response.text.encode('utf8'))
    Time = np.array([i["channels"][0]["value"] for i in req_result["data"]])
    # print(Time)

    data = np.array([TD, Rain, RH, WS, WD, Time])
    Y = np.transpose(data)
    return Y
