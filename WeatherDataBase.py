import numpy as np
import json
import requests
import datetime as dt

y1 = dt.date.today().year
m1 = dt.date.today().month
d1 = dt.date.today().day-1
headers = {'Authorization': 'ApiToken f058958a-d8bd-47cc-95d7-7ecf98610e47'}
#d2 = dt.date.today().day- i
#url = "https://api.ims.gov.il/v1/envista/stations/178" - info about Tel Aviv station
#TDurl = "https://api.ims.gov.il/v1/envista/stations/178/data/7?from="+str(y1)+"/"+str(m1)+"/"+str(d2)+"&to="+str(y1)+"/"+str(m1)+"/"+str(d1)

#Tempatures
TDurl = "https://api.ims.gov.il/v1/envista/stations/178/data/7/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
response = requests.request("GET", TDurl, headers=headers)
req_result = json.loads(response.text.encode('utf8'))
TD = np.array([i["channels"][0]["value"] for i in req_result["data"]])
#print(TD)

#Rain %
Rainurl = "https://api.ims.gov.il/v1/envista/stations/178/data/1/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
response = requests.request("GET", Rainurl, headers=headers)
req_result = json.loads(response.text.encode('utf8'))
Rain = np.array([i["channels"][0]["value"] for i in req_result["data"]])
#(Rain)

#Wind speed
WSurl = "https://api.ims.gov.il/v1/envista/stations/178/data/4/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
response = requests.request("GET", WSurl, headers=headers)
req_result = json.loads(response.text.encode('utf8'))
WS = np.array([i["channels"][0]["value"] for i in req_result["data"]])
#print(WS)

#Wind diraction
WDurl = "https://api.ims.gov.il/v1/envista/stations/178/data/5/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
response = requests.request("GET", WDurl, headers=headers)
req_result = json.loads(response.text.encode('utf8'))
WD = np.array([i["channels"][0]["value"] for i in req_result["data"]])
#print(WD)

RHurl = "https://api.ims.gov.il/v1/envista/stations/178/data/8/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
response = requests.request("GET", RHurl, headers=headers)
req_result = json.loads(response.text.encode('utf8'))
RH = np.array([i["channels"][0]["value"] for i in req_result["data"]])
#print(RH)

Timeurl = "https://api.ims.gov.il/v1/envista/stations/178/data/13/daily/"+str(y1)+"/"+str(m1)+"/"+str(d1)
response = requests.request("GET", Timeurl, headers=headers)
req_result = json.loads(response.text.encode('utf8'))
Time = np.array([i["channels"][0]["value"] for i in req_result["data"]])
#print(Time)

data = np.array([TD, Rain, RH, WS, WD, Time])
x = np.transpose(data)
print(x)
