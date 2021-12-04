import requests

payload = {'age': '58.0', 'sex':'0.0', 'cp': '1.0', 'trestbps':'120.0', 'chol':'165.0', 'fbs': '0.0', 'restecg': '1.0', 'thalach':'134.0', 'exang': '0.0', 'oldpeak': '0.0', 'slope':'1.0', 'ca': '0.0', 'thal':'2.0'}
r = requests.post("http://127.0.0.1:5000/predict", json=payload)
print(r)