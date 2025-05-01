# ⚛️ Quantum Explainable AI Dashboard

Welcome! This is an interactive dashboard to simulate and compare **credit risk assessment** using both classical and quantum models.

You can perform **interventional inference** (`do(X=1)`) and visually explore how different factors like credit limit, age, and repayment history affect the likelihood of credit default — using either:
- A classical backend with empirical distributions, or
- A quantum circuit-based Bayesian Network using PennyLane

---

## ⚗️ Features

- Classical model (`groupby` + joint probabilities)
- Quantum backend with RY-encoded qubit rotations
- Visual delta plots to show changes in `P(Default=1)`
- Powered by Streamlit, PennyLane, XGBoost, and pandas

---

## 📁 Dataset

Based on the [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

**Features used:**
- `LIMIT_BAL` – credit limit (income proxy)
- `Age`       – age group
- `PAY_AMT1`  – recent repayment amount
- `Default`   – binary target variable (0/1)

---

## 💻 Running Locally

```bash
git clone https://github.com/escelctron/qbn_app.git
cd qbn_app
pip install -r requirements.txt
streamlit run qbn_dashboard.py
```
---

## 🧑‍💻 About the Developer

Developed by **Pranav Sanghadia** — exploring the intersection of Quantum Computing and Explainable AI to bring transparency to financial, healthcare, and other domain's decision systems.

🎓 Currently pursuing a Master of Research in Quantum Computing (M.Res) at [Capitol Technology University](https://www.captechu.edu/degrees-and-programs/masters-degrees/quantum-computing-mres)


🔗 [Connect on LinkedIn](https://www.linkedin.com/in/sanghadia)

---

## 📝 License

MIT License
