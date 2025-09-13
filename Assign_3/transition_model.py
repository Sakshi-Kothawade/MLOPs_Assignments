# transition_model.py
from mlflow.tracking import MlflowClient

client = MlflowClient()

client.transition_model_version_stage(
    name="IrisRandomForestModel",
    version=1,
    stage="Production"
)
print("Model moved to Production.")
