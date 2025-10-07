# ============================================
# MLOps Assignment: Explainability + Fairness
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# -----------------------------
# STEP 1: Load and preprocess data
# -----------------------------
print("📥 Loading dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

data = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)

# Drop missing rows
data.dropna(inplace=True)

# Encode categorical columns
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Split features and target
X = data.drop("income", axis=1)
y = data["income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# STEP 2: Train the model
# -----------------------------
print("🤖 Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy: {acc:.2f}")

# -----------------------------
# STEP 3: Explainability with SHAP
# -----------------------------
print("🔍 Applying SHAP explainability...")
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

# Global feature importance plot
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.close()

# -----------------------------
# SHAP Single Prediction Fix
# -----------------------------
import shap

# Pick one sample
sample = X_test.iloc[[0]]

# Create force plot (HTML)
force_plot = shap.force_plot(
    explainer.expected_value, 
    shap_values[0, :], 
    sample
)

# Save to HTML file
shap.save_html("shap_single.html", force_plot)
print("🌐 SHAP single prediction saved: shap_single.html")


print("📊 SHAP plots saved: shap_summary.png (global) and shap_single.png (local)")

# -----------------------------
# STEP 4: Local Explainability with LIME (Optional)
# -----------------------------
print("💡 Applying LIME for one sample...")
lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=["<=50K", ">50K"],
    mode='classification'
)

i = 0  # explain first sample
exp = lime_explainer.explain_instance(
    data_row=X_test.iloc[i],
    predict_fn=model.predict_proba
)
exp.save_to_file('lime_explanation.html')
print("🌐 LIME explanation saved: lime_explanation.html")

# -----------------------------
# STEP 5: Fairness Audit (Gender)
# -----------------------------
print("⚖️ Auditing model fairness by gender...")

# In the dataset: 0 = Female, 1 = Male (after label encoding)
X_test_copy = X_test.copy()
X_test_copy['pred'] = y_pred

male_preds = X_test_copy[X_test['sex'] == 1]['pred']
female_preds = X_test_copy[X_test['sex'] == 0]['pred']

male_positive_rate = male_preds.mean()
female_positive_rate = female_preds.mean()

print("\n🔸 Fairness Audit Results:")
print(f"Male positive prediction rate:   {male_positive_rate:.2f}")
print(f"Female positive prediction rate: {female_positive_rate:.2f}")
print(f"Difference: {abs(male_positive_rate - female_positive_rate):.2f}")

if abs(male_positive_rate - female_positive_rate) > 0.1:
    print("⚠️ Potential fairness issue detected.")
else:
    print("✅ Model appears roughly fair by gender (basic check).")

print("\n✅ Assignment Completed!")
print("➡ Check the generated files: shap_summary.png, shap_single.png, lime_explanation.html")
