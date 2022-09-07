import requests 

resp = requests.post("https://cropera.herokuapp.com/predict", files={'file': open('0a6983a5-895e-4e68-9edb-88adf79211e9___RS_Early.B 9072.JPG', 'rb')})

print(resp.text)