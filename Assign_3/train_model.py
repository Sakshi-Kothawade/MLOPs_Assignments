import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score

# ✅ Set tracking server URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ✅ Optional: create/choose experiment
mlflow.set_experiment("iris_rf_experiment")

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ✅ Define input example (optional, suppresses warning)
input_example = pd.DataFrame(X_train[:5], columns=[f"feature_{i}" for i in range(X.shape[1])])

# Start and log run
with mlflow.start_run() as run:
    acc = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "rf_model", input_example=input_example)
    
    print("Run ID:", run.info.run_id)
