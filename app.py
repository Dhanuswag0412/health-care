import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="centered",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_or_train_model(csv_path: str):
    """Load model.pkl if present, otherwise train on the CSV."""
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        # Try loading a saved scaler/encoders alongside the model
        with open("preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    except FileNotFoundError:
        pass  # train from scratch

    df = pd.read_csv(csv_path)

    # ── Preprocessing ────────────────────────────────────────────────────────
    df = df.drop(columns=["id"])
    df = df[df["gender"] != "Other"]          # only 1 row
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preprocessor = {"scaler": scaler, "encoders": encoders, "feature_cols": list(X.columns)}
    return model, preprocessor


def encode_input(data: dict, preprocessor: dict) -> np.ndarray:
    encoders  = preprocessor["encoders"]
    scaler    = preprocessor["scaler"]
    feat_cols = preprocessor["feature_cols"]

    row = {}
    for col in feat_cols:
        val = data[col]
        if col in encoders:
            val = encoders[col].transform([val])[0]
        row[col] = val

    X = pd.DataFrame([row])[feat_cols]
    return scaler.transform(X)


# ── Load model ────────────────────────────────────────────────────────────────
CSV_PATH = "healthcare-dataset-stroke-data.csv"   # change path if needed

try:
    model, preprocessor = load_or_train_model(CSV_PATH)
    model_ready = True
except Exception as e:
    model_ready = False
    model_error = str(e)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧠 Stroke Risk Predictor")
st.markdown(
    "Fill in the patient details below and click **Predict** to estimate stroke risk."
)

if not model_ready:
    st.error(
        f"⚠️ Could not load or train the model.\n\n"
        f"Make sure **{CSV_PATH}** is in the same folder as this app.\n\n"
        f"Error: `{model_error}`"
    )
    st.stop()

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    gender         = st.selectbox("Gender", ["Male", "Female"])
    age            = st.slider("Age", 1, 100, 50)
    hypertension   = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
    heart_disease  = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
    ever_married   = st.selectbox("Ever Married", ["Yes", "No"])

with col2:
    work_type      = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose    = st.number_input("Avg Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=0.1)
    bmi            = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Stroke Risk", use_container_width=True, type="primary"):
    patient = {
        "gender":            gender,
        "age":               age,
        "hypertension":      hypertension,
        "heart_disease":     heart_disease,
        "ever_married":      ever_married,
        "work_type":         work_type,
        "Residence_type":    residence_type,
        "avg_glucose_level": avg_glucose,
        "bmi":               bmi,
        "smoking_status":    smoking_status,
    }

    try:
        X_input = encode_input(patient, preprocessor)
        pred    = model.predict(X_input)[0]
        proba   = model.predict_proba(X_input)[0][1] * 100

        if pred == 1:
            st.error(f"⚠️ **High Stroke Risk** — Estimated probability: **{proba:.1f}%**")
            st.markdown(
                "> This patient shows elevated risk factors. "
                "Please consult a neurologist or physician promptly."
            )
        else:
            st.success(f"✅ **Low Stroke Risk** — Estimated probability: **{proba:.1f}%**")
            st.markdown(
                "> Risk appears low based on the provided inputs. "
                "Regular health check-ups are still recommended."
            )

        # ── Feature summary ──────────────────────────────────────────────────
        with st.expander("📋 Input Summary"):
            summary = {
                "Gender": gender,
                "Age": age,
                "Hypertension": "Yes" if hypertension else "No",
                "Heart Disease": "Yes" if heart_disease else "No",
                "Ever Married": ever_married,
                "Work Type": work_type,
                "Residence Type": residence_type,
                "Avg Glucose (mg/dL)": avg_glucose,
                "BMI": bmi,
                "Smoking Status": smoking_status,
            }
            st.table(pd.DataFrame(summary.items(), columns=["Feature", "Value"]))

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚕️ **Disclaimer:** This tool is for educational purposes only and is not a "
    "substitute for professional medical advice, diagnosis, or treatment."
)
