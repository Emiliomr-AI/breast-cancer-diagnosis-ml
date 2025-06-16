import gradio as gr
import pandas as pd
from joblib import load

# Load trained model and scaler
scaler = load("model/scaler.joblib")
model = load("model/rf_model.joblib")

# Feature names in order
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Example input taken from a real malignant tumor case
example_input = [
    17.99, 10.38, 122.8, 1001, 0.1184,
    0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019, 0.1622,
    0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

def predict_diagnosis(*inputs):
    try:
        input_df = pd.DataFrame([inputs], columns=feature_names)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        label = "🩸 Malignant (Cancer)" if prediction == 1 else "✅ Benign (No cancer)"
        return f"🔍 Prediction: {label}\n\n📊 Confidence:\n - Malignant: {proba[1]:.2%}\n - Benign: {proba[0]:.2%}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Gradio interface
inputs = [gr.Number(label=col, precision=4) for col in feature_names]

iface = gr.Interface(
    fn=predict_diagnosis,
    inputs=inputs,
    outputs="text",
    title="🧬 Breast Cancer Diagnosis",
    description="Enter tumor cell features to predict malignancy",
    examples=[example_input]
)


if __name__ == "__main__":
    iface.launch()
