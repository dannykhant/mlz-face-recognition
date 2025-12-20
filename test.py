import requests

url = 'http://localhost:8000/predict'

request = {
    "url": "http://bit.ly/4j4Y0Uo"
}

result = requests.post(url, json=request).json()
print(result)
