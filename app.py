import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
import statsmodels.api as sm
from pathlib import Path
import plotly.express as px

# ==== Config ====
TARGET_TRANSFORM = "log"
MODEL_PATH = Path("model.joblib")
META_PATH = Path("model_metadata.json")

# ==== Helpers ====
def inverse_target(yhat: float) -> float:
    if TARGET_TRANSFORM == "log1p":
        return np.expm1(yhat)
    if TARGET_TRANSFORM == "log":
        return np.exp(yhat)
    return yhat

def encode_cat_vars(x: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(
        x,
        columns=x.select_dtypes(include=["object", "category"]).columns.tolist(),
        drop_first=True,
    )

# ==== Load model ====
model, model_cols = None, None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

if META_PATH.exists():
    meta = json.loads(META_PATH.read_text())
    TARGET_TRANSFORM = meta.get("target_transform", TARGET_TRANSFORM)
    model_cols = meta.get("features", None)

# ==== Sidebar ====
st.sidebar.image("https://img.icons8.com/color/96/000000/car.png", use_container_width=True)
st.sidebar.title("🚗 Used Car Price Predictor")
page = st.sidebar.radio("Select page", ["Prediction", "History"])

# ==== Main UI ====
if page == "Prediction":
    st.markdown(
        "<h2 style='text-align: center; color:#2E86C1;'>Used Car Price Prediction</h2>", 
        unsafe_allow_html=True
    )

    st.subheader("🔧 Enter Car Information")

    # Form với 2 cột
    with st.form("car_form"):
        col1, col2 = st.columns(2)

        with col1:
            location = st.selectbox("Location", ["Ha Noi","Ho Chi Minh","Da Nang","Hai Phong","Quang Ninh","Nghe An","Phu Tho","Can Tho","Hung Yen","Dong Nai","Thanh Hoa","Khanh Hoa"])
            kilometers = st.number_input("Kilometers Driven", 0, 2_000_000, 50000, 1000)
            fuel = st.selectbox("Fuel Type", ["Petrol","Diesel"])
            transmission = st.selectbox("Transmission", ["Manual","Automatic"])
            owner = st.selectbox("Owner Type", ["First","Second","Third","Fourth & Above"])

        with col2:
            mileage = st.number_input("Mileage (km/l)", 0.0, 60.0, 20.0, 0.1)
            power = st.number_input("Max Power (bhp)", 30.0, 700.0, 82.0, 1.0)
            seats = st.number_input("Seats", 2, 9, 5, 1)
            age = st.number_input("Age of Car (Years)", 1, 35, 8, 1)
            brand_class = st.selectbox("Brand Class", ["Low","Mid","High"])

        submit = st.form_submit_button("🚀 Predict")

    # Prediction Result ở dưới form
        st.subheader("📊 Prediction Result")
        if submit:
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

            if model is not None and hasattr(model, "model"):
                if "Kilometers_Driven_log" in model.model.exog_names:
                    row["Kilometers_Driven_log"] = np.log1p(row["Kilometers_Driven"])

            row_enc = encode_cat_vars(row).apply(pd.to_numeric, errors="coerce")

            if model is not None and hasattr(model, "model"):
                model_cols = model.model.exog_names
            if model_cols:
                if "const" in model_cols and "const" not in row_enc.columns:
                    row_enc = sm.add_constant(row_enc, has_constant="add")
                elif "const" not in model_cols and "const" in row_enc.columns:
                    row_enc = row_enc.drop(columns="const")
                row_enc = row_enc.reindex(columns=model_cols, fill_value=0.0)

            if model is not None:
                yhat = float(model.predict(row_enc))
                price = inverse_target(yhat)

                # Display result nicely
                st.metric("💰 Estimated Price", f"{int(price):,}M VND")

                # Save history (inputs + Predicted Price (M), bỏ Kilometers_Driven_log nếu có)
                to_save = row.copy()
                to_save = to_save.drop(columns=["Kilometers_Driven_log"], errors="ignore")
                to_save["Predicted Price (M)"] = int(price)
                
                # Khởi tạo history với đúng schema ngay từ đầu
                if "history" not in st.session_state or st.session_state["history"].empty:
                    # (đưa Predicted Price (M) xuống cuối cho gọn)
                    cols = [c for c in to_save.columns if c != "Predicted Price (M)"] + ["Predicted Price (M)"]
                    st.session_state["history"] = pd.DataFrame(columns=cols)
                
                # Căn cột rồi mới nối
                to_save = to_save.reindex(columns=st.session_state["history"].columns)
                st.session_state["history"] = pd.concat(
                    [st.session_state["history"], to_save], ignore_index=True
                )


                # Demo histogram (with simulated reference values)
                demo_df = pd.DataFrame({"Price (M)": np.random.normal(int(price), int(price)*0.15, 200)})
                fig = px.histogram(demo_df, x="Price (M)", nbins=30, title="Reference Price Distribution", color_discrete_sequence=["#2E86C1"])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Model not loaded.")

elif page == "History":
    st.subheader("📜 Prediction History")
    if "history" in st.session_state and not st.session_state["history"].empty:
        st.dataframe(st.session_state["history"])
        st.download_button("⬇️ Download CSV", st.session_state["history"].to_csv(index=False), "history.csv", "text/csv")
    else:
        st.info("No predictions yet.")










