# load_model.py
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models:/IrisRandomForestModel/Production")
print("Model loaded from registry.")
