import pickle
import numpy as np
import streamlit as st

# ----------------- Page config -----------------
st.set_page_config(
    page_title="Auto MPG Predictor",
    page_icon="🚗",
    layout="centered"
)

# ----------------- Custom CSS -----------------
st.markdown(
    """
    <style>
        /* Main background */
        .stApp {
            background: radial-gradient(circle at top left, #1e293b, #020617);
            color: #e5e7eb;
            font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Card-like containers */
        .prediction-card {
            background: linear-gradient(135deg, #020617aa, #0f172aaa);
            border-radius: 18px;
            padding: 1.5rem 1.8rem;
            border: 1px solid #334155;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.85);
        }

        .section-card {
            background: rgba(15, 23, 42, 0.92);
            border-radius: 16px;
            padding: 1.2rem 1.5rem;
            border: 1px solid #1f2937;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
        }

        /* Title styling */
        .app-title {
            font-size: 2.3rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #e5e7eb;
        }

        .highlight {
            background: linear-gradient(120deg, #22c55e, #0ea5e9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            font-size: 0.95rem;
            color: #9ca3af;
        }

        /* Metric styling */
        .big-metric {
            font-size: 2.1rem;
            font-weight: 700;
            color: #22c55e;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #9ca3af;
        }

        /* Slider label tweak */
        .stSlider label {
            font-weight: 600 !important;
            color: #e5e7eb !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617, #020617);
            border-right: 1px solid #1f2937;
        }

        /* Buttons */
        .stButton > button {
            width: 100%;
            border-radius: 999px;
            border: 1px solid #22c55e55;
            background: linear-gradient(120deg, #22c55e, #16a34a);
            color: white;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            box-shadow: 0 12px 30px rgba(34, 197, 94, 0.35);
        }
        .stButton > button:hover {
            border-color: #bbf7d0;
            transform: translateY(-1px);
        }

        .footer-text {
            font-size: 0.75rem;
            color: #6b7280;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Load model -----------------
@st.cache_resource
def load_model():
    try:
        with open("LinearRegression.pkl", "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, str(e)


model, model_error = load_model()

# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown("### ⚙️ Input Controls")
    st.markdown(
        "Tweak the sliders to simulate different car configurations and see how "
        "**MPG (fuel efficiency)** changes in real time."
    )

    horsepower = st.slider(
        "Engine Horsepower",
        min_value=40.0,
        max_value=250.0,
        value=130.0,
        step=1.0,
        help="Higher horsepower usually means more power but lower fuel efficiency."
    )

    weight = st.slider(
        "Vehicle Weight (lbs)",
        min_value=1500.0,
        max_value=5000.0,
        value=3200.0,
        step=50.0,
        help="Heavier cars generally consume more fuel."
    )

    acceleration = st.slider(
        "0–60 mph Acceleration (seconds)",
        min_value=8.0,
        max_value=25.0,
        value=15.0,
        step=0.5,
        help="Time taken to reach 60 mph — performance indicator."
    )

    st.markdown("---")
    st.markdown(
        "<span class='footer-text'>💡 Tip: Combine high horsepower, low acceleration time, "
        "and high weight to simulate sports or muscle cars.</span>",
        unsafe_allow_html=True,
    )

# ----------------- Main Layout -----------------
st.markdown(
    "<div class='app-title'>🚗 Auto <span class='highlight'>MPG Predictor</span></div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='subtitle'>A clean, production-style interface to estimate a car's "
    "fuel efficiency (miles per gallon) using a trained Linear Regression model.</p>",
    unsafe_allow_html=True,
)

st.markdown("")  # spacing

col_main, col_side = st.columns([2.2, 1])

with col_main:
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)

    st.markdown("#### 🎯 Prediction Panel")
    st.markdown(
        "Provide your car specifications in the sidebar and click **Predict MPG** "
        "to estimate its fuel efficiency."
    )

    if model_error:
        st.error(
            "⚠️ Could not load the model file `LinearRegression.pkl`.\n\n"
            f"**Details:** `{model_error}`\n\n"
            "Make sure the `.pkl` file is in the same folder as this `app.py` script."
        )
    else:
        predict_btn = st.button("🔮 Predict MPG")

        if predict_btn:
            # Arrange features in the SAME ORDER as training: [horsepower, acceleration, weight]
            features = np.array([[horsepower, acceleration, weight]])

            try:
                pred_mpg = model.predict(features)[0]

                st.markdown("---")
                st.markdown("##### Result")
                cols = st.columns([1.2, 1])
                with cols[0]:
                    st.markdown("<div class='metric-label'>Estimated Fuel Efficiency</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='big-metric'>{pred_mpg:.2f} MPG</div>",
                        unsafe_allow_html=True,
                    )
                with cols[1]:
                    # Simple derived metric: L/100km approx = 235.215 / MPG
                    if pred_mpg > 0:
                        l_per_100km = 235.215 / pred_mpg
                        st.metric(label="Approx. L/100 km", value=f"{l_per_100km:.2f}")
                    else:
                        st.metric(label="Approx. L/100 km", value="N/A")

                st.markdown("---")
                st.markdown("##### Interpretation")
                if pred_mpg >= 30:
                    st.success(
                        "🌱 **Great efficiency!** This configuration suggests a very fuel-efficient vehicle."
                    )
                elif 20 <= pred_mpg < 30:
                    st.info(
                        "⚖️ **Moderate efficiency.** A balanced mix of performance and fuel economy."
                    )
                else:
                    st.warning(
                        "⛽ **Low efficiency.** This setup looks more performance or weight oriented than fuel-efficient."
                    )

            except Exception as e:
                st.error(
                    "An error occurred while predicting MPG. "
                    f"Please check the model file and feature order.\n\n`{e}`"
                )

    st.markdown("</div>", unsafe_allow_html=True)

with col_side:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("#### 🔍 Model Overview")
    st.markdown(
        """
        - **Algorithm:** Linear Regression  
        - **Target:** MPG (miles per gallon)  
        - **Features used:**
          - Horsepower  
          - Acceleration  
          - Weight
        """
    )

    st.markdown("---")
    st.markdown("#### 📘 How it works")
    st.markdown(
        """
        1. You set the input values in the sidebar  
        2. The app passes them to the trained model  
        3. The model returns the estimated MPG  
        4. We also show a rough fuel usage in L/100 km
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("ℹ️ About this project"):
    st.write(
        """
        This app is powered by a Linear Regression model trained on the classic Auto MPG dataset.
        It demonstrates a full ML workflow: data cleaning, outlier removal, model training, and
        packaging the model into a user-friendly web interface with Streamlit.
        """
    )
