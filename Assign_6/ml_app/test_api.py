import requests

# Example input for your model (Iris dataset)
data = {"features": [5.1, 3.5, 1.4, 0.2]}

# Send POST request to your running Flask API
response = requests.post("http://127.0.0.1:5000/predict", json=data)

# Print the prediction returned by the API
print(response.json())
