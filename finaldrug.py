import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 🔧 EXPECTED SCHEMA (lean)
# ---------------------------
EXPECTED_BASE_COLS = ["Age", "Gender", "Education", "Country", "Ethnicity"]
POSSIBLE_DRUG_COLS = [
    "Alcohol", "Amphet", "Amyl", "Benzos", "Caffeine", "Cannabis", "Chocolate",
    "Coke", "Crack", "Ecstasy", "Heroin", "Ketamine", "Legalh", "LSD",
    "Meth", "Mushrooms", "Nicotine", "Semer", "VSA"
]
VALID_CLASSES = {"CL0","CL1","CL2","CL3","CL4","CL5","CL6"}

# ---------------------------
# 🧠 App
# ---------------------------
st.set_page_config(page_title="Custom Drug Classifier", layout="wide")
st.title("🧠 Multi-Drug Use Classifier")
st.markdown("Upload your drug dataset and select any drug (e.g., Alcohol, Meth, Cannabis) to analyze usage patterns.")

with st.sidebar.expander("ℹ️ Expected data format"):
    st.write("**Your CSV should contain:**")
    st.markdown(
        "- Base columns:\n  " + ", ".join(f"`{c}`" for c in EXPECTED_BASE_COLS) +
        "\n- **At least one** drug column like: " + ", ".join(f"`{c}`" for c in POSSIBLE_DRUG_COLS) +
        "\n- Drug values must be one of: " + ", ".join(sorted(VALID_CLASSES))
    )

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

def is_drug_column(df, col):
    unique_vals = set(df[col].dropna().astype(str).unique())
    return bool(unique_vals) and unique_vals.issubset(VALID_CLASSES)

def schema_error_popup(missing_cols=None, no_drug=False, bad_drug_cols=None):
    msg_lines = [":rotating_light: **Schema mismatch** – please fix your CSV and re-upload."]
    msg_lines.append("**Make sure your data contains these fields:**")
    msg_lines.append("- Base columns: " + ", ".join(f"`{c}`" for c in EXPECTED_BASE_COLS))
    msg_lines.append("- **At least one** drug column like: " + ", ".join(f"`{c}`" for c in POSSIBLE_DRUG_COLS))
    msg_lines.append("- Drug values must be one of: " + ", ".join(sorted(VALID_CLASSES)))

    if missing_cols:
        msg_lines.append(f"\n**Missing base columns:** {', '.join('`'+c+'`' for c in missing_cols)}")
    if no_drug:
        msg_lines.append("\n**No valid drug column found** (a drug column must only contain CL0–CL6).")
    if bad_drug_cols:
        pretty = ", ".join(f"`{c}`" for c in bad_drug_cols)
        msg_lines.append(f"\n**Invalid values detected** in {pretty} (must be only CL0–CL6).")

    st.error("\n".join(msg_lines))
    st.stop()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # ---------------------------
    # ✅ Schema validation
    # ---------------------------
    missing_base = [c for c in EXPECTED_BASE_COLS if c not in df.columns]

    candidate_drug_cols = [c for c in df.columns if c in POSSIBLE_DRUG_COLS]
    valid_drug_cols, bad_value_drug_cols = [], []

    scan_cols = candidate_drug_cols if candidate_drug_cols else list(df.columns)
    for c in scan_cols:
        try:
            if is_drug_column(df, c):
                valid_drug_cols.append(c)
            else:
                if c in POSSIBLE_DRUG_COLS:
                    bad_value_drug_cols.append(c)
        except Exception:
            pass

    if missing_base or (not valid_drug_cols) or bad_value_drug_cols:
        schema_error_popup(
            missing_cols=missing_base if missing_base else None,
            no_drug=(len(valid_drug_cols) == 0),
            bad_drug_cols=bad_value_drug_cols if bad_value_drug_cols else None
        )

    # ---------------------------
    # ✅ If schema OK → continue
    # ---------------------------
    drug_cols = valid_drug_cols
    selected_drug = st.sidebar.selectbox("🔬 Select a Drug to Analyze", drug_cols)

    df[f"{selected_drug}_Binary"] = df[selected_drug].astype(str).apply(
        lambda x: 0 if x in ['CL0', 'CL1', 'CL2'] else 1
    )

    working_df = df.drop(columns=[selected_drug])

    for col in working_df.columns:
        if working_df[col].dtype == 'object':
            le = LabelEncoder()
            working_df[col] = le.fit_transform(working_df[col].astype(str))

    X = working_df.drop(columns=[f"{selected_drug}_Binary"])
    y = working_df[f"{selected_drug}_Binary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(C=0.5, solver='liblinear'),
        "SVM (RBF Kernel)": SVC(C=1.0, kernel='rbf', gamma='scale'),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, criterion='entropy'),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "y_pred": y_pred,
            "report": classification_report(y_test, y_pred, output_dict=True),
            "confusion": confusion_matrix(y_test, y_pred)
        }

    st.subheader(f"📊 Accuracy Comparison for: {selected_drug}")
    acc_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy (%)": [round(v["accuracy"] * 100, 2) for v in results.values()]
    })
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=acc_df, x="Model", y="Accuracy (%)", palette="viridis", ax=ax)
    ax.set_ylim(0, 100)
    for index, row in acc_df.iterrows():
        ax.text(index, row["Accuracy (%)"] + 1, f"{row['Accuracy (%)']}%", ha='center', fontweight='bold')
    ax.set_title(f"Model Accuracy on Predicting {selected_drug} Use")
    st.pyplot(fig)

    selected_model = st.selectbox("🔍 Select a model to view report", list(results.keys()))

    st.subheader(f"📋 Classification Report - {selected_model}")
    report_df = pd.DataFrame(results[selected_model]["report"]).transpose()
    st.dataframe(
        report_df.style.format({
            'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.0f}'
        }).background_gradient(cmap='Blues'),
        use_container_width=True
    )

    st.subheader(f"🧩 Confusion Matrix - {selected_model}")
    cm = results[selected_model]["confusion"]
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_xticklabels(['Non-User', 'User'])
    ax_cm.set_yticklabels(['Non-User', 'User'], rotation=0)
    st.pyplot(fig_cm)

    user_count = int(sum(results[selected_model]['y_pred']))
    non_user_count = int(len(results[selected_model]['y_pred']) - user_count)
    st.subheader("📈 Final Verdict")
    if user_count > non_user_count:
        st.success(f"More predicted as **USERS** ({user_count}) than non-users ({non_user_count}).")
    elif non_user_count > user_count:
        st.success(f"More predicted as **NON-USERS** ({non_user_count}) than users ({user_count}).")
    else:
        st.info("Equal number of users and non-users predicted.")
