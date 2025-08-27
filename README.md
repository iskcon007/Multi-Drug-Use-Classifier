# 💊 Drug X Predictor – Multi-Drug Classifier

## 📌 Overview
Drug X Predictor is a **Streamlit web application** that analyzes drug usage patterns from survey data.  
It predicts whether individuals are likely **Users** or **Non-Users** of a selected drug using multiple machine learning models.

---

## 🚀 Features
- 📂 Upload your own dataset (CSV format)
- 🔬 Select a drug column (CL0–CL6 format)
- 🧠 Models included: Logistic Regression, SVM, Decision Tree, Random Forest
- 📊 Accuracy comparison across models
- 🧩 Classification report + confusion matrix
- 📈 Final verdict on users vs non-users

---

## 🛠 Tech Stack
- **Frontend/UI** → Streamlit
- **ML Models** → Scikit-learn
- **Visualization** → Matplotlib, Seaborn
- **Data Handling** → Pandas, NumPy

---

## 📦 Run Locally
```bash
pip install -r requirements.txt
streamlit run finaldrug.py
