# register_model.py
import mlflow

run_id = "<PASTE_RUN_ID_HERE>"  # From training output

result = mlflow.register_model(
    f"runs:/{run_id}/rf_model",
    "IrisRandomForestModel"
)

print("Model registered:", result)
