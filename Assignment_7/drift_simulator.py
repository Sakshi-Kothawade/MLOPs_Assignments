# drift_simulator.py
import requests
import numpy as np
import time
import json

MODEL_API_URL = "http://localhost:5000/predict"

def generate_request_data(is_drifted=False):
    """Generates feature data for an inference request."""
    if not is_drifted:
        # Phase 1: Normal Conditions (e.g., typical Pune climate)
        # Mean temperature: 25°C, std dev: 3
        temp = np.random.normal(loc=25, scale=3)
        # Mean rainfall: 120mm, std dev: 20
        rain = np.random.normal(loc=120, scale=20)
    else:
        # Phase 2: Drifted Conditions (e.g., a heatwave period)
        # Mean temperature has drifted upwards to 35°C!
        print("--- DRIFT ACTIVATED ---")
        temp = np.random.normal(loc=35, scale=4)
        # Rainfall remains the same for this simulation
        rain = np.random.normal(loc=120, scale=20)

    return {"temperature": round(temp, 2), "rainfall": round(rain, 2)}


if __name__ == "__main__":
    print("Starting simulation...")
    headers = {'Content-Type': 'application/json'}
    drift_after_requests = 60 # Introduce drift after 60 requests (approx 2 mins)
    request_count = 0

    while True:
        try:
            # Determine if we should send drifted data
            is_drift_active = request_count > drift_after_requests

            # Generate data and send request
            payload = generate_request_data(is_drifted=is_drift_active)
            response = requests.post(MODEL_API_URL, data=json.dumps(payload), headers=headers)
            response.raise_for_status() # Raise an exception for bad status codes

            print(f"Sent: {payload}, Received: {response.json()}")

            request_count += 1
            time.sleep(2) # Wait 2 seconds between requests

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to the model API: {e}")
            time.sleep(5)