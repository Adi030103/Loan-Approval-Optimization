#  Lending Club Default Prediction & Loan Approval Optimization
**Course Project: Predictive Modeling + Offline Reinforcement Learning**

---

##  Project Overview

This project builds two complementary AI systems for credit risk management using **Lending Club’s public loan dataset** (2007–2018):

1. **Deep Learning Classifier (Supervised)**
   - Predicts the probability of **loan default** from borrower features.
   - Evaluated using **AUC** and **F1-score**.

2. **Offline Reinforcement Learning Agent (Contextual Bandit)**
   - Learns a **loan approval policy** that maximizes **expected profit** directly.
   - Evaluated using **Estimated Policy Value (EPV)** — mean expected reward per applicant.



## Dataset

We used the official **Lending Club Accepted Loans** dataset:
```
accepted_2007_to_2018Q4.csv.gz
```

- Total records: ~2.26M loans
- Target column: `loan_status`
- Binary mapping:
  - `Fully Paid` → 0
  - `Charged Off / Default` → 1

---

##  Model 1 — Predictive Deep Learning Model

### **Goal:** Predict risk of default from applicant features.

**Pipeline:**
1. Cleaned missing values, parsed dates, converted percentage fields.
2. Engineered features:
   - `int_rate_num`, `revol_util_num`, `term_months`, `emp_len_yrs`, `credit_hist_mths`
3. Selected **leakage-free** features (only data known at decision time).
4. Trained **PyTorch MLP** (3 hidden layers, ReLU + Dropout).

**Results:**
| Metric | Value |
|:-------|:------|
| AUC | 0.708 |
| F1 (t=0.50) | 0.37 |

**Interpretation:**
- Model ranks borrowers by risk ~71% accurately.
- Balanced precision–recall for early-stage screening.

---

## Model 2 — Offline Reinforcement Learning Agent

### **Goal:** Learn a direct approve/deny policy to maximize profit.

| Component | Definition |
|:-----------|:------------|
| **State (s)** | Applicant features (88-dim vector) |
| **Action (a)** | {0: Deny, 1: Approve} |
| **Reward (r)** | `+ loan_amnt * int_rate` if paid; `- loan_amnt` if default; `0` if denied |

**Algorithm:**
- Trained **Direct Method (XGBoost regressor)** to predict expected reward.
- Policy rule: approve if predicted reward > 0.

**Results:**

| Policy | EPV (mean reward/applicant) | Approval Rate |
|:-------|------------------------------|----------------|
| DL @ 0.50 | 1,536.6 | 57.3% |
| DL @ 0.95 (profit-optimal) | 3,553.4 | 100% |
| RL (XGB-DM) | **3,494.7** | 96.9% |



## 🧮 Insights

- **DL model:** strong ranking, moderate classification.
- **RL policy:** directly profit-optimized, higher EPV.
- **High thresholds (≈0.9–0.95)** balance risk vs reward.
- **Disagreements:** RL approves some high-risk/high-return loans DL rejects.



## 👤 Author

**Aditya Srivastava**  
B.E. COE — Thapar Institute of Engineering & Technology  


