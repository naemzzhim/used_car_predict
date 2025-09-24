import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
import statsmodels.api as sm
from pathlib import Path

# ==== Cấu hình ====
TARGET_TRANSFORM = "log"   # bạn train Price_log = np.log(Price)
MODEL_PATH = Path(r"C:\Users\ADMIN\Desktop\DATN\Used_Car_Price_Predict\model.joblib")
META_PATH = Path(r"C:\Users\ADMIN\Desktop\DATN\Used_Car_Price_Predict\model_metadata.json")

# ==== Helpers ====
def inverse_target(yhat: float) -> float:
    if TARGET_TRANSFORM == "log1p":
        return np.expm1(yhat)
    if TARGET_TRANSFORM == "log":
        return np.exp(yhat)
    return yhat

def encode_cat_vars(x: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode các cột object/category, drop_first=True"""
    return pd.get_dummies(
        x,
        columns=x.select_dtypes(include=["object", "category"]).columns.tolist(),
        drop_first=True,
    )

# ==== Load model + metadata ====
model = None
model_cols = None

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Model loaded thành công")
    except Exception as e:
        st.error(f"Lỗi load model: {e}")

if META_PATH.exists():
    try:
        meta = json.loads(META_PATH.read_text())
        TARGET_TRANSFORM = meta.get("target_transform", TARGET_TRANSFORM)
        model_cols = meta.get("features", None)
    except Exception as e:
        st.warning(f"Không đọc được metadata: {e}")

# ==== Giao diện web ====
st.title("🚗 Used Car Price Predictor")
st.write("Hi! This is my first Streamlit website, try it on! 🎉")
st.header("Type your car's information")

with st.form("car_form"):
    location = st.selectbox("Location", ["Ha Noi", "Ho Chi Minh", "Da Nang", "Hai Phong", "Can Tho",
    "Thanh Hoa", "Nghe An", "Hue", "Quang Ninh", "Hai Duong",
    "Bac Ninh", "Nam Dinh", "Thai Binh", "Ha Nam", "Ninh Binh",
    "Lao Cai", "Lang Son", "Bac Giang", "Phu Tho", "Thai Nguyen"])
    kilometers = st.number_input("Kilometers Driven", min_value=0, max_value=2_000_000, value=50000, step=1000)
    fuel = st.selectbox("Fuel Type", ["Petrol","Diesel"])
    transmission = st.selectbox("Transmission", ["Manual","Automatic"])
    owner = st.selectbox("Owner Type", ["First","Second","Third","Fourth & Above"])
    mileage = st.number_input("Mileage (km/l)", min_value=0.0, max_value=60.0, value=20.0, step=0.1)
    power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=700.0, value=82.0, step=1.0)
    seats = st.number_input("Seats", min_value=2, max_value=9, value=5, step=1)
    age = st.number_input("Age of car (Years)", min_value=1, max_value=35, value=8, step=1)
    brand_class = st.selectbox("Brand Class", ["Low","Mid","High"])
    submit = st.form_submit_button("Click here to Predict Price")

if submit:
    # 1. Gom input thành DataFrame
    row = pd.DataFrame([{
        "Location": location,
        "Kilometers_Driven": kilometers,
        "Fuel_Type": fuel,
        "Transmission": transmission,
        "Owner_Type": owner,
        "Mileage": mileage,
        "Power": power,
        "Seats": seats,
        "Ageofcar": age,
        "Brand_Class": brand_class,
    }])

    # 2. Sinh thêm feature log nếu model cần
    if model is not None and hasattr(model, "model"):
        if "Kilometers_Driven_log" in model.model.exog_names:
            row["Kilometers_Driven_log"] = np.log1p(row["Kilometers_Driven"])

    # 3. Encode categorical
    row_enc = encode_cat_vars(row).apply(pd.to_numeric, errors="coerce")

    # 4. Đồng bộ cột theo exog_names
    if model is not None and hasattr(model, "model"):
        model_cols = model.model.exog_names
    if model_cols:
        if "const" in model_cols and "const" not in row_enc.columns:
            row_enc = sm.add_constant(row_enc, has_constant="add")
        elif "const" not in model_cols and "const" in row_enc.columns:
            row_enc = row_enc.drop(columns="const")
        row_enc = row_enc.reindex(columns=model_cols, fill_value=0.0)

    # 5. Dự đoán
    if model is not None:
        try:
            yhat = float(model.predict(row_enc))
            price = inverse_target(yhat)
            st.success(f"💰 Estimated Price: {int(price):,} triệu VND")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Debug columns:", row_enc.columns.tolist())
    else:
        st.warning("⚠️ Model chưa được train hoặc load.")
