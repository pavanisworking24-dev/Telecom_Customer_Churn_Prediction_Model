# 📡 Telecom Customer Churn Predictor

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn — combining a full ML pipeline (EDA → feature engineering → model selection → business-cost threshold tuning → SHAP interpretability) with a live Streamlit app for real-time and batch predictions.

## 🚀 Live Demo

👉 **[Launch the Streamlit App](https://customer-churn-predictor-website.streamlit.app/)**

![ROC AUC Curves of all models](images/ROC_AUC Curves.png)

---

## 💡 Business Impact

Customer churn is one of the most expensive problems in telecom — acquiring a new customer costs far more than retaining an existing one. This project doesn't just classify churners; it optimizes for the decision telecom teams actually have to make.

- **91.98% recall** at the selected decision threshold — catching the vast majority of customers who are about to leave, before they leave.
- **Threshold tuned against an explicit business cost model**: a false-positive retention offer costs ~$100, while a missed churner costs ~$500. The default 0.5 threshold was replaced with **0.35** specifically to minimize total business cost, not just to maximize accuracy.
- **SHAP-based explainability** turns model output into concrete retention actions — e.g., prioritizing outreach for month-to-month, fiber-optic, first-3-months customers, rather than a black-box risk score.

---

## 🧠 Key Finding: Why the "best" model on paper wasn't the model we shipped

XGBoost scored highest during cross-validation, but **Logistic Regression generalized better on the held-out test set** (ROC-AUC 0.8392 vs. XGBoost's 0.8316, with a much higher final recall). The gap traced back to XGBoost overfitting to the synthetic distribution introduced by SMOTE oversampling. Rather than defaulting to the more complex algorithm because it looked stronger on paper, the simpler, better-generalizing model was selected — and the threshold was then independently tuned to the business cost function.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.7424 | 0.5094 | 0.7968 | 0.6215 | **0.8392** |
| Decision Tree | 0.7310 | 0.4950 | 0.6631 | 0.5669 | 0.7893 |
| Random Forest | 0.7750 | 0.5697 | 0.6230 | 0.5951 | 0.8249 |
| XGBoost | 0.7750 | 0.5711 | 0.6123 | 0.5910 | 0.8316 |

![Performnace comparision of all models](images/model_performance_comparision.png)


**Final selected model — Logistic Regression, threshold-tuned:**
- ROC-AUC: `0.8392`
- Optimal Threshold: `0.35` (business-cost-minimizing, not default 0.5)
- Final Recall: `0.9198`
- Final Precision: `0.4514`
- Final F1: `0.6056`

![Confusion Matrix of final selected logestic model](images/confusion_matrix.png)

> The threshold was intentionally lowered from the default 0.5 to 0.35 to maximize recall — in a retention context, catching a churner is worth far more than avoiding a false alarm.

---
![Feature importance of final selected logestic model](images/Feature_importance.png)
## 🧪 ML Pipeline

| Step | Description |
|---|---|
| 1 | Data Loading & Exploratory Data Analysis (EDA) |
| 2 | Feature Engineering (21 raw → 51 engineered features) |
| 3 | Train-Test Split |
| 4 | Encoding & Scaling |
| 5 | Class Imbalance Handling (SMOTE, applied only to training data to avoid leakage) |
| 6 | Feature Selection (L1 / Lasso regularization → 32 selected features) |
| 7 | Model Training (Logistic Regression · Decision Tree · Random Forest · XGBoost) |
| 8 | Cross-Validation & Hyperparameter Tuning (RandomizedSearchCV, 5-fold stratified) |
| 9 | Model Comparison & Generalization Check |
| 10 | Threshold Tuning (Business Cost Optimization: FP=$100, FN=$500) |
| 11 | ROC-AUC · Precision-Recall · Business Cost Curves |
| 12 | SHAP Interpretability |


---

## 🗂️ Repository Structure

```
├── notebooks/                 # Full ML pipeline notebook (EDA → modeling → SHAP)
├── PowerPoint Presentation/    # Project summary slides
├── application.py             # Streamlit app — single entry point
├── churn_model.pkl            # Saved Logistic Regression model
├── scaler.pkl                 # Fitted StandardScaler
├── all_features.pkl           # Full encoded feature list (pre-selection)
├── selected_features.pkl      # L1-selected feature subset used by the model
├── threshold.pkl              # Business-cost-optimal decision threshold (0.35)
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ⚙️ Tech Stack

- **Language:** Python 3.12
- **ML:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Interpretability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Web App:** Streamlit
- **Serialization:** Joblib

---

## 🛠️ Run Locally

```bash
# Clone the repository
git clone https://github.com/pavanisworking24-dev/Telecom_Customer_Churn_Prediction_Model.git
cd Telecom_Customer_Churn_Prediction_Model

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run application.py
```

---

## 📊 Dataset

**IBM Telco Customer Churn Dataset**

- 7,043 customer records
- 21 raw features → 51 after feature engineering → 32 after L1 selection
- Target: `Churn` (Yes / No), ~26.5% churn rate

---

## 🔑 Key Learnings

- **Cross-validation performance isn't the same as generalization.** The model that wins on CV metrics can still overfit — always validate on a genuinely held-out test set before shipping.
- **The "right" threshold depends on the cost of being wrong, not just accuracy.** Optimizing for a business cost function produced a very different (and more useful) threshold than the default 0.5.
- **Correct handling of edge cases matters**: zero-tenure customers were explicitly set to $0 TotalCharges rather than median-imputed, preserving the real tenure–billing relationship instead of introducing noise.

---

## 👤 Author

**Venkata Pavan Kumar** — [GitHub](https://github.com/pavanisworking24-dev) · [LinkedIn](https://www.linkedin.com/in/venkatapavankumarpagadala)

---

## 📄 License

This project is available under the [MIT License](LICENSE).

> ⭐ If you found this project helpful, consider giving it a star!
