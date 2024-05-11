import requests

inputs = {"input": "Racial Discrimination cases"}
response = requests.post("http://localhost:8080/chat/invoke", json=inputs)
data = response.json()
print(data)