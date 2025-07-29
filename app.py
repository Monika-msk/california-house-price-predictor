import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from fpdf import FPDF
import io
import os
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Feature Info ---
feature_names = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]
feature_desc = {
    "MedInc": "Median income in block group (10,000s USD)",
    "HouseAge": "Median house age (years)",
    "AveRooms": "Average rooms per household",
    "AveBedrms": "Average bedrooms per household",
    "Population": "Population in block group",
    "AveOccup": "Average occupants per household",
    "Latitude": "Latitude (geo coordinate)",
    "Longitude": "Longitude (geo coordinate)"
}

# --- Theme Selector ---
THEMES = {
    "Light Mode": {
        "primaryColor": "#1565C0",
        "backgroundColor": "#F7FBFF",
        "secondaryBackgroundColor": "#E8F1FD",
        "textColor": "#212946"
    },
    "Dark Mode": {
        "primaryColor": "#F3B601",
        "backgroundColor": "#181C25",
        "secondaryBackgroundColor": "#1E2230",
        "textColor": "#F5F5F5"
    }
}

theme_choice = st.sidebar.selectbox(
    "Choose App Theme",
    list(THEMES.keys()),
    index=0,
    key="app_theme_selector"
)
theme = THEMES[theme_choice]

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {theme['backgroundColor']} !important;
        color: {theme['textColor']} !important;
    }}
    .stSidebarContent {{
        background-color: {theme['secondaryBackgroundColor']} !important;
    }}
    .stButton>button, 
    .stDownloadButton>button, 
    .stFormSubmitButton>button {{
        background-color: {theme['primaryColor']} !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: bold !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="California House Price Predictor", page_icon="🏡")

# Sidebar content
st.sidebar.title("About")
st.sidebar.markdown(
    """
    **California House Price Predictor**

    This app predicts median house prices in California using state-of-the-art machine learning models. 

    
    - **Explore the Data:** View dataset overview, feature descriptions, sample data, histograms, and pairplots.
    - **Model Evaluation:** Review model metrics (MAE, MSE, RMSE, R²) on a hold-out test set.
    - **Make Predictions:** Enter custom values for all features. Select either Linear Regression or Random Forest.
    - **Understand Predictions:** For Random Forest, see feature importance and SHAP-based impact for your prediction.
    - **Download Results:** Get a PDF report or CSV file of your input and prediction.
    - **Customize Look:** Use the sidebar to switch between light and dark themes.

    ---
    *Created by Monika S Kumar, 2025*
    """
)


# --- TITLE ---
st.markdown(
    "<h1 style='white-space:nowrap;'>🏡 California House Price Predictor</h1>",
    unsafe_allow_html=True
)
st.divider()

# --- DATASET OVERVIEW AND EDA ---
st.markdown("### Dataset Overview")
california = fetch_california_housing(as_frame=True)
df = california.frame

st.markdown(f"**Shape:** {df.shape[0]:,} rows, {df.shape[1]} columns")
with st.expander("ℹ️ Feature Descriptions and Sample Data"):
    for key, desc in feature_desc.items():
        st.markdown(f"- **{key}**: {desc}")
    st.dataframe(df.head())

# Histogram
with st.expander("🧭 Explore Feature Distributions (Histograms)"):
    selected_col = st.selectbox("Choose feature for histogram", df.columns, key="hist_feat")
    fig, ax = plt.subplots()
    ax.hist(df[selected_col], bins=30, color=theme['primaryColor'], edgecolor='black')
    ax.set_title(f"Histogram of {selected_col}")
    ax.set_xlabel(selected_col)
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Pairplot


with st.expander("📊 Pairplot of All Features "):
    show_pair = st.checkbox("Show pairplot of ALL features")
    if show_pair:
        # Use a random sample to avoid performance problems
        sample_df = df.sample(n=1000, random_state=42)
        fig2 = sns.pairplot(sample_df, corner=True)
        st.pyplot(fig2)

# --- MODEL EVALUATION METRICS ---
# Split data for evaluation
X = df[feature_names]
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Load Pretrained Models and Scaler ---
scaler = joblib.load('scaler.joblib')
linreg_model = joblib.load('linreg_model.joblib')
rf_model = joblib.load('rf_model.joblib')
rf_importances = np.load('rf_importances.npy')

# Scale test set
X_test_scaled = scaler.transform(X_test)

# Predict with both models on test set
y_pred_linreg = linreg_model.predict(X_test_scaled) * 100000
y_pred_rf = rf_model.predict(X_test_scaled) * 100000
y_test_true = y_test * 100000

def get_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, pred)
    return mae, mse, rmse, r2

metrics_linreg = get_metrics(y_test_true, y_pred_linreg)
metrics_rf = get_metrics(y_test_true, y_pred_rf)

with st.expander("📈 Model Evaluation Metrics (on Test Set)"):
    st.markdown("**Evaluated on a 20% hold-out test set**")
    st.markdown("""
| Metric       | Linear Regression | Random Forest |
|:-------------|------------------:|--------------:|
| MAE          | {:,.2f}           | {:,.2f}       |
| MSE          | {:,.2f}           | {:,.2f}       |
| RMSE         | {:,.2f}           | {:,.2f}       |
| R²           | {:.4f}            | {:.4f}        |
    """.format(
        metrics_linreg[0], metrics_rf[0],
        metrics_linreg[1], metrics_rf[1],
        metrics_linreg[2], metrics_rf[2],
        metrics_linreg[3], metrics_rf[3]
    )
    )

st.divider()
# --- MODEL SELECTION ---
model_choice = st.selectbox(
    "Select the prediction model:",
    ["Linear Regression", "Random Forest"],
    key="model_choice"
)

# --- FORM FOR USER INPUT ---
with st.form("input_form"):
    st.subheader("Enter feature values:")
    user_input = []
    for feat in feature_names:
        val = st.number_input(
            f"{feat}", value=1.0, format="%.2f", help=feature_desc[feat]
        )
        user_input.append(val)
    submitted = st.form_submit_button("Predict")

pred = None

def create_pdf_report(features, prediction, chart_fig=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=16, style='B')
    pdf.cell(0, 10, "California House Price Prediction Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Helvetica", size=12, style='B')
    pdf.cell(0, 10, "Input Features:", ln=True)
    pdf.set_font("Helvetica", size=12)
    for feat, val in features.items():
        pdf.cell(0, 8, f"{feat}: {val}", ln=True)

    pdf.ln(8)
    pdf.set_font("Helvetica", size=14, style='B')
    pdf.cell(0, 10, f"Predicted Median House Value: ${prediction:.2f}", ln=True)

    if chart_fig:
        chart_fig.savefig("temp_chart.png", format='PNG', bbox_inches='tight')
        pdf.ln(10)
        pdf.image("temp_chart.png", x=20, w=170)
        os.remove("temp_chart.png")

    pdf.output("prediction_report.pdf")
    with open("prediction_report.pdf", "rb") as f:
        pdf_bytes = f.read()
    return pdf_bytes

if submitted:
    invalid = False
    for feat, value in zip(feature_names, user_input):
        if value < 0 and feat not in ["Latitude", "Longitude"]:
            st.warning(f"{feat} should not be negative.")
            invalid = True
    if not (32 <= user_input[6] <= 42):
        st.warning("Latitude should be between 32 and 42 (California).")
        invalid = True
    if not (-125 <= user_input[7] <= -113):
        st.warning("Longitude should be between -125 and -113 (California).")
        invalid = True

    if not invalid:
        X_scaled = scaler.transform([user_input])

        if model_choice == "Linear Regression":
            pred = linreg_model.predict(X_scaled)[0] * 100000
            st.success(f"Linear Regression Predicted Median House Value: ${pred:.2f}")

        elif model_choice == "Random Forest":
            pred = rf_model.predict(X_scaled)[0] * 100000
            st.success(f"Random Forest Predicted Median House Value: ${pred:.2f}")

            st.divider()
            st.subheader("🔎 Feature Importance (Random Forest):")
            fig, ax = plt.subplots()
            ax.barh(feature_names, rf_importances, color="teal")
            ax.set_xlabel("Importance Score")
            ax.set_title("Random Forest Feature Importance")
            st.pyplot(fig)

            st.divider()
            st.subheader("🔍 Feature Impact for This Prediction (SHAP Values)")
            background = np.load('background_scaled.npy')
            explainer = shap.Explainer(rf_model, background)
            shap_values = explainer(X_scaled)
            shap_values_np = shap_values.values[0]
            fig2, ax2 = plt.subplots()
            ax2.barh(feature_names, shap_values_np, color="orange")
            ax2.set_title("SHAP Value per Feature")
            ax2.set_xlabel("SHAP Value")
            st.pyplot(fig2)

            # PDF Report Download
            data_dict = {fn: val for fn, val in zip(feature_names, user_input)}
            pdf_report = create_pdf_report(data_dict, pred, chart_fig=fig2)
            st.download_button(
                label="Download Prediction Report (PDF)",
                data=pdf_report,
                file_name="house_price_prediction_report.pdf",
                mime="application/pdf",
            )

        st.divider()
        # CSV Download
        data = {fn: [val] for fn, val in zip(feature_names, user_input)}
        data['Predicted Value'] = [pred]
        df_out = pd.DataFrame(data)
        csv = df_out.to_csv(index=False)
        st.download_button("Download prediction as CSV", csv, "prediction.csv", "text/csv")
    else:
        st.info("Please correct the highlighted inputs before predicting.")
else:
    st.divider()
    st.info("Fill the inputs and click Predict to see the result and download predictions.")

st.markdown("---")
st.caption("Made by Monika S Kumar, 2025")
