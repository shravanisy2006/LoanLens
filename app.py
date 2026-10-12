from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "loan.csv"
MODEL_DIR = BASE_DIR / "models"
ASSET_DIR = BASE_DIR / "assets"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = MODEL_DIR / "model_comparison.csv"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

st.set_page_config(
    page_title="LoanLens AI | Risk Intelligence",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# THEME
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
.stApp { background: #0b1120; color: #e6edf7; font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: #111b2e; border-right: 1px solid #26344c; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1500px; }
.hero { padding: 1.6rem; border: 1px solid #263c56; border-radius: 20px; background: linear-gradient(120deg,#13243b,#102b35); margin-bottom: 1.2rem; }
.hero h1 { font-family: 'Space Grotesk',sans-serif; font-size: 2.5rem; margin-bottom: .3rem; color: #f5f8ff; }
.hero p { color: #a9bed2; margin-bottom: 0; line-height: 1.6; }
.metric-card { background: #121d30; border: 1px solid #293a52; border-radius: 15px; padding: 18px; min-height: 108px; }
.metric-label { color: #9eb1c8; font-size: .78rem; letter-spacing: .7px; }
.metric-value { color: #71e0cc; font-family: 'Space Grotesk',sans-serif; font-size: 1.55rem; font-weight: 700; margin-top: 7px; overflow-wrap: anywhere; }
.section-note { color: #9eb1c8; font-size: .9rem; }
div[data-testid="stForm"] { border: 1px solid #293a52; border-radius: 16px; padding: 20px; }
.stButton > button, .stFormSubmitButton > button { background: #19a995; color: white; border: none; border-radius: 9px; font-weight: 600; min-height: 42px; }
.stButton > button:hover, .stFormSubmitButton > button:hover { background: #138b7b; color: white; }
div[data-testid="stExpander"] { border: 1px solid #293a52; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------
required_files = [DATA_PATH, MODEL_PATH, METRICS_PATH, METADATA_PATH]
missing_files = [str(path) for path in required_files if not path.exists()]
if missing_files:
    st.error("Required project files are missing. Run `python train_model.py` first.")
    for path in missing_files:
        st.code(path)
    st.stop()

@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)

@st.cache_data
def load_metadata(path_string):
    with open(path_string, "r", encoding="utf-8") as file:
        return json.load(file)

@st.cache_data
def load_metrics(path_string):
    return pd.read_csv(path_string)

@st.cache_data
def load_data(path_string):
    data = pd.read_csv(path_string, low_memory=False)
    data = data.dropna(axis=1, how="all")
    data = data[data["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    data["default"] = (data["loan_status"] == "Charged Off").astype(int)
    for col in ["int_rate", "revol_util"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col].astype(str).str.replace("%", "", regex=False), errors="coerce")
    for col in ["loan_amnt", "annual_inc", "dti", "installment", "revol_bal", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "total_acc"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data

model = load_model(str(MODEL_PATH))
metadata = load_metadata(str(METADATA_PATH))
metrics = load_metrics(str(METRICS_PATH))
df = load_data(str(DATA_PATH))

FEATURES = metadata["features"]
NUMERIC_FEATURES = metadata["numeric_features"]
CATEGORICAL_FEATURES = metadata["categorical_features"]
DECISION_THRESHOLD = float(metadata.get("decision_threshold", 0.5))
BEST_MODEL_NAME = metadata.get("best_model", "Saved model")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def metric_card(label, value):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def predict_probability(row):
    frame = pd.DataFrame([{feature: row[feature] for feature in FEATURES}], columns=FEATURES)
    return float(model.predict_proba(frame)[0, 1])

def probability_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        number={"suffix": "%", "font": {"size": 40, "color": "#71e0cc"}},
        title={"text": "Estimated Default Probability"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#19a995"}, "bgcolor": "#121d30", "bordercolor": "#293a52",
            "steps": [
                {"range": [0, 15], "color": "#153b3a"},
                {"range": [15, 30], "color": "#514322"},
                {"range": [30, 100], "color": "#4a2634"},
            ],
        },
    ))
    fig.update_layout(template="plotly_dark", height=310, margin=dict(l=20, r=20, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def mode_or_first(series, fallback):
    values = series.dropna().astype(str)
    if values.empty:
        return fallback
    modes = values.mode()
    return str(modes.iloc[0]) if not modes.empty else str(values.iloc[0])

def safe_numeric_default(column, fallback=0.0):
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else float(fallback)

def category_options(column, fallback):
    values = sorted(df[column].dropna().astype(str).unique().tolist())
    return values if values else [fallback]

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 💠 LoanLens AI")
    st.caption("LOAN RISK INTELLIGENCE")
    st.divider()
    page = st.radio("NAVIGATION", ["Overview", "Predict & Simulate", "3D Loan Explorer", "Model Lab"])
    st.divider()
    st.caption("MACHINE LEARNING")
    st.write(f"Selected model: **{BEST_MODEL_NAME}**")
    st.caption(f"Saved decision threshold: {DECISION_THRESHOLD:.4f}")
    st.caption("Educational prototype · Not a lending decision system")

# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------
if page == "Overview":
    total = len(df)
    defaults = int(df["default"].sum())
    paid = total - defaults
    default_rate = defaults / total * 100 if total else 0
    hero("LoanLens AI", "Explore loan risk. Understand model predictions. Compare scenarios.")
    st.caption("A machine-learning research prototype built using historical LendingClub loan records.")
    cols = st.columns(4)
    summary = [("RESOLVED LOANS", f"{total:,}"), ("FULLY PAID", f"{paid:,}"), ("CHARGED OFF", f"{defaults:,}"), ("HISTORICAL DEFAULT RATE", f"{default_rate:.2f}%")]
    for col, (label, value) in zip(cols, summary):
        with col:
            metric_card(label, value)
    st.write("")
    left, right = st.columns(2)
    with left:
        st.subheader("Loan Outcomes")
        outcome = df["loan_status"].value_counts().rename_axis("Outcome").reset_index(name="Loans")
        fig = px.pie(outcome, names="Outcome", values="Loans", hole=0.65, color_discrete_sequence=["#7187ff", "#fa725e"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=370)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Default Rate by Grade")
        grade_stats = df.groupby("grade", observed=True)["default"].agg(["mean", "count"]).reset_index()
        grade_stats["Default Rate (%)"] = grade_stats["mean"] * 100
        fig = px.bar(grade_stats, x="grade", y="Default Rate (%)", color="Default Rate (%)", color_continuous_scale=["#19a995", "#71e0cc", "#fa725e"], hover_data={"count": True, "Default Rate (%)": ":.2f"}, labels={"grade": "Loan Grade"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, height=370)
        st.plotly_chart(fig, use_container_width=True)
    st.info("Use the sidebar to predict risk, explore historical loans in 3D, or compare model performance.")

# --------------------------------------------------
# PAGE 2: PREDICTION AND SIMULATION
# --------------------------------------------------
elif page == "Predict & Simulate":
    hero("Risk Scenario Studio", "Enter borrower details to estimate default probability, then explore an alternative loan amount.")
    st.warning("This is an experimental estimate based on historical LendingClub data, not calibrated financial advice or a lending decision.")

    # Defaults based on the historical dataset. Advanced fields are optional to the user,
    # but all model-required features are supplied to the pipeline.
    defaults = {col: safe_numeric_default(col) for col in NUMERIC_FEATURES}
    defaults.update({col: mode_or_first(df[col], "Unknown") for col in CATEGORICAL_FEATURES})

    with st.form("prediction_form"):
        st.subheader("Essential details")
        a, b, c = st.columns(3)
        with a:
            loan_amnt = st.number_input("Loan amount ($)", min_value=500.0, max_value=40000.0, value=float(np.clip(defaults.get("loan_amnt", 10000), 500, 40000)), step=500.0)
            annual_inc = st.number_input("Annual income ($)", min_value=1000.0, max_value=1000000.0, value=float(np.clip(defaults.get("annual_inc", 60000), 1000, 1000000)), step=1000.0)
        with b:
            dti = st.slider("Debt-to-income ratio (%)", 0.0, 50.0, float(np.clip(defaults.get("dti", 18), 0, 50)), 0.5)
            int_rate = st.slider("Interest rate (%)", 5.0, 30.0, float(np.clip(defaults.get("int_rate", 12), 5, 30)), 0.25)
        with c:
            grade_options = category_options("grade", "C")
            grade_default = defaults.get("grade", "C")
            grade = st.selectbox("Loan grade", grade_options, index=grade_options.index(grade_default) if grade_default in grade_options else 0)
            term_options = category_options("term", " 36 months")
            term_default = defaults.get("term", term_options[0])
            term = st.selectbox("Loan term", term_options, index=term_options.index(term_default) if term_default in term_options else 0)

        with st.expander("Advanced Details"):
            st.caption("These fields are initialized from typical values in the dataset. Adjust them if you know the borrower's information.")
            num_cols = st.columns(3)
            advanced_numeric = {}
            numeric_bounds = {
                "installment": (0.0, 3000.0, 10.0),
                "delinq_2yrs": (0.0, 50.0, 1.0),
                "inq_last_6mths": (0.0, 50.0, 1.0),
                "open_acc": (0.0, 100.0, 1.0),
                "pub_rec": (0.0, 30.0, 1.0),
                "revol_bal": (0.0, 1000000.0, 500.0),
                "total_acc": (0.0, 150.0, 1.0),
                "revol_util": (0.0, 150.0, 1.0),
            }
            for index, col in enumerate(NUMERIC_FEATURES):
                if col in {"loan_amnt", "annual_inc", "dti", "int_rate"}:
                    continue
                lo, hi, step = numeric_bounds.get(col, (0.0, 1000000.0, 1.0))
                val = float(np.clip(defaults.get(col, 0), lo, hi))
                with num_cols[index % 3]:
                    advanced_numeric[col] = st.number_input(col.replace("_", " ").title(), min_value=lo, max_value=hi, value=val, step=step, key=f"advanced_{col}")
            cat_cols = st.columns(2)
            advanced_categorical = {}
            for index, col in enumerate(CATEGORICAL_FEATURES):
                if col in {"grade", "term"}:
                    continue
                opts = category_options(col, str(defaults.get(col, "Unknown")))
                selected = defaults.get(col, opts[0])
                with cat_cols[index % 2]:
                    advanced_categorical[col] = st.selectbox(col.replace("_", " ").title(), opts, index=opts.index(selected) if selected in opts else 0, key=f"advanced_cat_{col}")

        submitted = st.form_submit_button("Estimate Default Probability", use_container_width=True)

    if submitted:
        row = defaults.copy()
        row.update({
            "loan_amnt": loan_amnt,
            "annual_inc": annual_inc,
            "dti": dti,
            "int_rate": int_rate,
            "grade": grade,
            "term": term,
        })
        row.update(advanced_numeric)
        row.update(advanced_categorical)
        # The training pipeline expects numeric percentage values, not strings with %.
        input_row = {feature: row[feature] for feature in FEATURES}
        probability = predict_probability(input_row)
        st.session_state["last_profile"] = input_row
        st.session_state["last_probability"] = probability

    if "last_probability" in st.session_state and "last_profile" in st.session_state:
        row = st.session_state["last_profile"]
        probability = float(st.session_state["last_probability"])
        predicted_default = probability >= DECISION_THRESHOLD
        st.divider()
        st.subheader("Prediction Results")
        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(probability_gauge(probability), use_container_width=True)
        with right:
            st.metric("Estimated default probability", f"{probability:.1%}")
            st.write(f"**Model:** {BEST_MODEL_NAME}")
            st.write(f"**Decision threshold:** {DECISION_THRESHOLD:.4f} ({DECISION_THRESHOLD:.1%})")
            if predicted_default:
                st.error("Model classification: Potential default")
            else:
                st.success("Model classification: Below the selected default threshold")
            st.caption("The threshold determines the model's binary classification. It does not change the probability itself or constitute a real lending decision.")

        st.subheader("What-If Simulator")
        st.write("Change the loan amount to compare an alternative scenario. Other inputs stay the same, and the installment is scaled proportionally as an approximation.")
        current_amount = float(row["loan_amnt"])
        alt_amount = st.slider("Alternative loan amount ($)", min_value=500, max_value=40000, value=int(np.clip(current_amount + 5000, 500, 40000)), step=500, key="alternate_amount")
        alt_row = row.copy()
        alt_row["loan_amnt"] = alt_amount
        if "installment" in FEATURES and current_amount > 0:
            alt_row["installment"] = float(row["installment"]) * alt_amount / current_amount
        alt_probability = predict_probability(alt_row)
        comparison = pd.DataFrame({"Scenario": ["Original", "Alternative"], "Default probability (%)": [probability * 100, alt_probability * 100]})
        fig = px.bar(comparison, x="Scenario", y="Default probability (%)", color="Scenario", text="Default probability (%)", color_discrete_sequence=["#19a995", "#7187ff"])
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, yaxis_range=[0, max(100, max(comparison["Default probability (%)"]) * 1.2)])
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Change in estimate: **{(alt_probability - probability) * 100:+.2f} percentage points**")
        st.caption("This is a model sensitivity comparison, not proof that changing the loan amount causes a change in real-world default risk.")

# --------------------------------------------------
# PAGE 3: 3D LOAN EXPLORER
# --------------------------------------------------
elif page == "3D Loan Explorer":
    hero("3D Loan Universe", "Explore historical loan records across three financial dimensions.")
    st.caption("Each point represents a historical loan. Colors represent the final recorded outcome, not a prediction for a new borrower.")
    available = df.dropna(subset=["loan_amnt", "annual_inc", "dti"]).copy()
    outcome_filter = st.radio("Outcomes", ["Both outcomes", "Fully Paid", "Charged Off"], horizontal=True)
    if outcome_filter != "Both outcomes":
        available = available[available["loan_status"] == outcome_filter]
    max_points = st.slider("Number of records to display", 500, 5000, 2000, step=500)
    sample = available.sample(n=min(max_points, len(available)), random_state=42) if len(available) else available
    if sample.empty:
        st.warning("No records are available for this selection.")
    else:
        fig = px.scatter_3d(sample, x="loan_amnt", y="annual_inc", z="dti", color="loan_status", hover_data=[col for col in ["grade", "int_rate", "purpose", "loan_status"] if col in sample.columns], color_discrete_map={"Fully Paid": "#71e0cc", "Charged Off": "#fa725e"}, opacity=0.75, labels={"loan_amnt": "Loan Amount ($)", "annual_inc": "Annual Income ($)", "dti": "Debt-to-Income Ratio (%)"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=680, scene=dict(bgcolor="#0b1120", xaxis_title="Loan Amount ($)", yaxis_title="Annual Income ($)", zaxis_title="DTI (%)"))
        st.plotly_chart(fig, use_container_width=True)
    st.info("Drag to rotate, scroll to zoom, and hover over points to inspect historical records.")

# --------------------------------------------------
# PAGE 4: MODEL LAB
# --------------------------------------------------
elif page == "Model Lab":
    hero("Model Intelligence Lab", "Compare tuned classifiers on the held-out test set.")
    st.caption(f"Selected model by training cross-validation: {BEST_MODEL_NAME}. Threshold tuned on out-of-fold training predictions.")
    preferred_metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Average Precision"]
    display_metrics = [col for col in preferred_metrics if col in metrics.columns]
    if "Model" not in metrics.columns or not display_metrics:
        st.warning("The saved model_comparison.csv does not contain the expected metrics. Please rerun train_model.py.")
    else:
        st.subheader("Model Performance Comparison")
        chart_metric = st.selectbox("Metric to visualize", display_metrics, index=display_metrics.index("F1 Score") if "F1 Score" in display_metrics else 0)
        fig = px.bar(metrics, x="Model", y=chart_metric, color="Model", text=chart_metric, color_discrete_sequence=["#71e0cc", "#7187ff", "#f6bd60"])
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, min(1.2, max(1.0, float(metrics[chart_metric].max()) * 1.15))], height=430, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Detailed Test Metrics")
        format_cols = {col: "{:.4f}" for col in display_metrics if pd.api.types.is_numeric_dtype(metrics[col])}
        if "Threshold" in metrics.columns:
            format_cols["Threshold"] = "{:.4f}"
        st.dataframe(metrics.style.format(format_cols), use_container_width=True, hide_index=True)
        if "Threshold" in metrics.columns:
            selected_row = metrics[metrics["Model"] == BEST_MODEL_NAME]
            if not selected_row.empty:
                st.info(f"Selected model threshold: {float(selected_row.iloc[0]['Threshold']):.4f}. The threshold was chosen using training out-of-fold predictions.")

    st.subheader("Confusion Matrix")
    best_matrix = ASSET_DIR / "best_model_confusion_matrix.png"
    if best_matrix.exists():
        st.image(str(best_matrix), caption=f"{BEST_MODEL_NAME} — held-out test set", use_container_width=True)
    else:
        old_matrix_files = ["logistic_regression_confusion_matrix.png", "random_forest_confusion_matrix.png", "decision_tree_confusion_matrix.png"]
        found = False
        for filename in old_matrix_files:
            path = ASSET_DIR / filename
            if path.exists():
                st.image(str(path), caption=filename.replace("_", " "), use_container_width=True)
                found = True
        if not found:
            st.caption("No saved confusion matrix found. Run train_model.py to generate it.")

    pr_path = ASSET_DIR / "precision_recall_curve.png"
    if pr_path.exists():
        with st.expander("View precision-recall curve"):
            st.image(str(pr_path), use_container_width=True)

    st.download_button("Download model comparison CSV", data=metrics.to_csv(index=False).encode("utf-8"), file_name="loanlens_model_comparison.csv", mime="text/csv")
    st.caption("Metrics are held-out test results for the tuned models. Performance may vary on future data; the historical dataset is not a guarantee of real-world performance.")

st.divider()
st.caption("LoanLens AI · LendingClub historical data · Educational project prototype")
