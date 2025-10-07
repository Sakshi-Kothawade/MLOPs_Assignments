# app.py (FastAPI version)
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, Summary, make_wsgi_app
import random
import time

# 1. Define Pydantic model for input data validation
class InferenceRequest(BaseModel):
    temperature: float = 25.0
    rainfall: float = 100.0

# 2. Create FastAPI App
app = FastAPI(
    title="KrishiConnect Model API",
    description="API for crop yield prediction with built-in monitoring.",
    version="1.0.0"
)

# 3. Define the same Prometheus Metrics
PREDICTIONS_TOTAL = Counter(
    'krishiconnect_predictions_total',
    'Total number of predictions made.'
)
INPUT_TEMPERATURE = Histogram(
    'krishiconnect_input_temperature_celsius',
    'Distribution of input temperature feature.'
)
INPUT_RAINFALL = Histogram(
    'krishiconnect_input_rainfall_mm',
    'Distribution of input rainfall feature.'
)
PREDICTION_CONFIDENCE = Gauge(
    'krishiconnect_prediction_confidence_score',
    'Confidence score of the last prediction.'
)
REQUEST_LATENCY = Summary(
    'krishiconnect_request_latency_seconds',
    'Time spent processing a request.'
)

# 4. Mount the /metrics endpoint
# This adds the Prometheus WSGI app to the FastAPI application
metrics_app = make_wsgi_app()
app.mount("/metrics", WSGIMiddleware(metrics_app))

# --- Prediction Endpoint (CHANGES ARE HERE) ---
@app.post("/predict")
# REMOVED the @REQUEST_LATENCY.time() decorator from here
async def predict(data: InferenceRequest):
    """
    Takes temperature and rainfall data and returns a predicted crop yield.
    """
    start_time = time.time() # 1. Record start time

    temperature = data.temperature
    rainfall = data.rainfall

    # --- Dummy Model Logic (same as before) ---
    yield_prediction = 100
    if temperature > 30 and rainfall > 80:
        yield_prediction += 20
    elif temperature < 20:
        yield_prediction -= 15
    confidence = random.uniform(0.85, 0.99)
    # --- End of Dummy Logic ---

    # --- Update metrics (same as before) ---
    PREDICTIONS_TOTAL.inc()
    INPUT_TEMPERATURE.observe(temperature)
    INPUT_RAINFALL.observe(rainfall)
    PREDICTION_CONFIDENCE.set(confidence)

    response_data = {
        "predicted_yield": yield_prediction,
        "confidence": confidence
    }

    # 2. Calculate latency and observe it before returning
    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency)

    return response_data

@app.get("/")
def root():
    return {
        "message": "KrishiConnect Model API is running.",
        "docs_url": "/docs",
        "metrics_url": "/metrics"
    }

# This part is for running the app directly with uvicorn for development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)