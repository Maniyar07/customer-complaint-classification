import streamlit as st
import pandas as pd
import requests
import json
from pathlib import Path
import os

# Page Config

st.set_page_config(
    page_title="Customer Complaint Classifier",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL

FLASK_API_URL = os.getenv("FLASK_API_URL", "http://127.0.0.1:5000")

# Local Paths for Reports and Charts

BASE_DIR = Path(__file__).resolve().parent

REPORT_PATH = BASE_DIR / "reports" / "03_best_classification_report.csv"
TOP_CATEGORIES_IMG = BASE_DIR / "reports" / "top_15_categories.png"
WORD_COUNT_IMG = BASE_DIR / "reports" / "complaint_word_count_distribution.png"
EXPERIMENT_FILE = BASE_DIR / "reports" / "03_data_feature_experiment_results.csv"
MISCLASSIFIED_FILE = BASE_DIR / "reports" / "03_misclassified_samples.csv"

# CSS

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0B1120 0%, #111827 55%, #0F172A 100%);
        color: #F8FAFC;
    }

    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #263244;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }
    
    .compact-hero {
        background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
        padding: 26px 24px 18px 24px;
        border-radius: 18px;
        border: 1px solid #334155;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        margin-top: 8px;
        margin-top: 10px;
        margin-bottom: 16px;
        
    }
    
    .app-title {
        font-size: 31px;
        font-weight: 900;
        line-height: 1.35;
        color: #F8FAFC;
        margin-bottom: 6px;
        padding-top: 4px;
    }

    .app-subtitle {
        font-size: 15px;
        color: #CBD5E1;
        line-height: 1.5;
    }

    .mini-metric-card {
        background: rgba(30, 41, 59, 0.78);
        padding: 13px 15px;
        border-radius: 15px;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 5px 16px rgba(0,0,0,0.18);
        min-height: 82px;
    }

    .mini-metric-label {
        color: #94A3B8;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .mini-metric-value {
        color: #38BDF8;
        font-size: 25px;
        font-weight: 900;
        margin-top: 5px;
    }

    .prediction-box {
        background: linear-gradient(135deg, #047857, #10B981);
        padding: 18px;
        border-radius: 18px;
        color: white;
        margin-top: 14px;
        text-align: center;
        box-shadow: 0 8px 22px rgba(16,185,129,0.25);
    }

    .prediction-title {
        font-size: 15px;
        font-weight: 700;
        color: #DCFCE7;
    }

    .prediction-value {
        font-size: 24px;
        font-weight: 900;
        margin-top: 6px;
    }

    .stTextArea textarea {
        border-radius: 14px;
        font-size: 15px;
        background-color: #111827;
        color: #F8FAFC;
        border: 1px solid #475569;
    }

    .stButton button {
        width: 100%;
        border-radius: 13px;
        height: 44px;
        font-size: 15px;
        font-weight: 800;
        background: linear-gradient(90deg, #2563EB, #06B6D4);
        color: white;
        border: none;
        box-shadow: 0 5px 15px rgba(37,99,235,0.25);
    }

    .stButton button:hover {
        background: linear-gradient(90deg, #1D4ED8, #0891B2);
        color: white;
        border: none;
    }

    h1, h2, h3 {
        color: #F8FAFC;
    }

    h2 {
        font-size: 1.45rem !important;
        margin-top: 0.4rem !important;
    }

    h3 {
        font-size: 1.08rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# API Helper Functions

def check_api_health():
    try:
        response = requests.get(f"{FLASK_API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def get_metrics():
    try:
        response = requests.get(f"{FLASK_API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception:
        return {}


def predict_complaint(complaint_text):
    try:
        response = requests.post(
            f"{FLASK_API_URL}/predict",
            json={"complaint_text": complaint_text},
            timeout=10
        )

        return response.json()

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def format_percent(value):
    try:
        value = float(value)
        if value <= 1:
            value = value * 100
        return f"{value:.2f}%"
    except Exception:
        return "N/A"


def get_metric_value(metrics_dict, possible_keys, default=0):
    cleaned_metrics = {}

    for key, value in metrics_dict.items():
        clean_key = str(key).strip().lower().replace(" ", "_")
        cleaned_metrics[clean_key] = value

    for key in possible_keys:
        clean_search_key = str(key).strip().lower().replace(" ", "_")
        if clean_search_key in cleaned_metrics:
            return cleaned_metrics[clean_search_key]

    return default

# Check Flask API

api_health = check_api_health()

if api_health is None:
    st.error("Flask backend API is not running.")
    st.info("First run this command in one terminal:")
    st.code("python app.py", language="bash")
    st.info("Then run Streamlit in another terminal:")
    st.code("streamlit run streamlit_app.py", language="bash")
    st.stop()

metrics = get_metrics()
PROJECT_CONFIG = {
    "text_version": "clean_text",
    "max_features": 30000,
    "min_samples_per_class": 50,
    "model_name": "TF-IDF + Logistic Regression"
}

# Metrics Values

accuracy = metrics.get("accuracy", 0)
macro_f1 = metrics.get("macro_f1", 0)
weighted_f1 = metrics.get("weighted_f1", 0)

training_rows = metrics.get("num_rows", 20874)
num_classes = metrics.get("num_classes", 8)

text_version = metrics.get(
    "text_version",
    PROJECT_CONFIG["text_version"]
)

max_features = metrics.get(
    "max_features",
    PROJECT_CONFIG["max_features"]
)

min_samples = metrics.get(
    "min_samples_per_class",
    PROJECT_CONFIG["min_samples_per_class"]
)

# Sidebar

with st.sidebar:
    st.markdown("## 📊 Project Info")
    st.markdown("**Project:** Customer Complaint Classification")
    st.markdown("**Backend:** Flask API")
    st.markdown("**Frontend:** Streamlit UI")
    st.markdown("**Model:** TF-IDF + Logistic Regression")

    st.divider()

    st.markdown("## 🚀 Best Experiment")
    st.markdown(f"**Text Version:** `{text_version}`")
    st.markdown(f"**Max Features:** `{max_features}`")
    st.markdown(f"**Min Samples/Class:** `{min_samples}`")
    st.markdown(f"**Training Rows:** `{training_rows}`")
    st.markdown(f"**Classes:** `{num_classes}`")

    st.divider()

    st.markdown("## 🔌 API Status")
    st.success("Flask API connected")


# Header

st.markdown(
    """
    <div class="compact-hero">
        <div class="app-title">📩 Customer Complaint Classification</div>
        <div class="app-subtitle">
            Streamlit frontend connected with Flask backend API for NLP-based complaint category prediction.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Metrics Cards

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="mini-metric-card">
            <div class="mini-metric-label">Accuracy</div>
            <div class="mini-metric-value">{format_percent(accuracy)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    st.markdown(
        f"""
        <div class="mini-metric-card">
            <div class="mini-metric-label">Macro F1</div>
            <div class="mini-metric-value">{format_percent(macro_f1)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:
    st.markdown(
        f"""
        <div class="mini-metric-card">
            <div class="mini-metric-label">Weighted F1</div>
            <div class="mini-metric-value">{format_percent(weighted_f1)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m4:
    st.markdown(
        f"""
        <div class="mini-metric-card">
            <div class="mini-metric-label">Training Rows</div>
            <div class="mini-metric-value">{training_rows}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Prediction Section

st.markdown("## 🔍 Predict Complaint Category")

sample_complaints = {
    "Debt collection issue":
        "I am receiving repeated calls from a debt collector for a debt that I do not recognize.",
    "Credit card issue":
        "My credit card was charged with an unknown transaction and the bank is not resolving my issue.",
    "Bank account issue":
        "Money was debited from my checking account without my permission.",
    "Mortgage issue":
        "My mortgage payment was applied incorrectly and I am being charged extra fees.",
    "Vehicle loan issue":
        "The lender is showing incorrect balance on my vehicle loan account.",
    "Student loan issue":
        "My student loan payment was not updated even after successful payment."
}

left_col, right_col = st.columns([2.2, 1])

with left_col:
    complaint_text = st.text_area(
        "Enter customer complaint text",
        height=125,
        placeholder="Example: Money was debited from my account without permission..."
    )

    predict_button = st.button("🔍 Predict Category")

with right_col:
    st.markdown("### Try sample")

    selected_sample = st.selectbox(
        "Select example",
        list(sample_complaints.keys())
    )

    use_sample = st.button("Use Sample")

    if use_sample:
        st.session_state["sample_text"] = sample_complaints[selected_sample]

    if "sample_text" in st.session_state:
        st.info(st.session_state["sample_text"])

if not complaint_text and "sample_text" in st.session_state:
    complaint_text = st.session_state["sample_text"]

if predict_button:
    if complaint_text.strip() == "":
        st.warning("Please enter complaint text before prediction.")
    else:
        result = predict_complaint(complaint_text)

        if result.get("status") == "success":
            predicted_category = result.get("predicted_category")

            st.markdown(
                f"""
                <div class="prediction-box">
                    <div class="prediction-title">Predicted Complaint Category</div>
                    <div class="prediction-value">{predicted_category}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            confidence_scores = result.get("confidence_scores", [])

            if confidence_scores:
                confidence_df = pd.DataFrame(confidence_scores)

                with st.expander("View prediction confidence"):
                    st.dataframe(
                        confidence_df[["category", "confidence_percent"]],
                        use_container_width=True,
                        hide_index=True
                    )

                    chart_df = confidence_df.set_index("category")["confidence"]
                    st.bar_chart(chart_df)

        else:
            st.error(result.get("message", "Prediction failed."))

st.divider()

# Reports and Charts Tabs

tab1, tab2, tab3 = st.tabs([
    "📋 Classification Report",
    "📊 EDA Charts",
    "📂 Extra Reports"
])

with tab1:
    st.markdown("### Classification Report")

    if REPORT_PATH.exists():
        report_df = pd.read_csv(REPORT_PATH)
        st.dataframe(report_df, use_container_width=True, hide_index=False)
    else:
        st.warning("03_best_classification_report.csv not found inside reports folder.")

with tab2:
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("### Top 15 Categories")
        if TOP_CATEGORIES_IMG.exists():
            st.image(str(TOP_CATEGORIES_IMG), use_container_width=True)
        else:
            st.warning("top_15_categories.png not found inside reports folder.")

    with chart_col2:
        st.markdown("### Word Count Distribution")
        if WORD_COUNT_IMG.exists():
            st.image(str(WORD_COUNT_IMG), use_container_width=True)
        else:
            st.warning("complaint_word_count_distribution.png not found inside reports folder.")

with tab3:
    extra_col1, extra_col2 = st.columns(2)

    with extra_col1:
        st.markdown("### Experiment Results")
        if EXPERIMENT_FILE.exists():
            experiment_df = pd.read_csv(EXPERIMENT_FILE)
            st.dataframe(experiment_df, use_container_width=True)
        else:
            st.info("Experiment results file not found.")

    with extra_col2:
        st.markdown("### Misclassified Samples")
        if MISCLASSIFIED_FILE.exists():
            misclassified_df = pd.read_csv(MISCLASSIFIED_FILE)
            st.dataframe(misclassified_df, use_container_width=True)
        else:
            st.info("Misclassified samples file not found.")

# Footer

st.divider()

st.markdown(
    """
    <div style='text-align:center; color:#94A3B8; padding:14px; font-size:14px;'>
        Flask Backend API + Streamlit Frontend UI | Customer Complaint Classification
    </div>
    """,
    unsafe_allow_html=True
)