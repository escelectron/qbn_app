---

```markdown
# 🔮 Quantum Explainable AI Dashboard

Welcome!!! This is an interactive dashboard to simulate and compare **credit risk assessment** using both classical and quantum models.

You can perform **interventional inference** (do(X=1)) and visually explore how different factors like credit limit, age, and repayment history affect the likelihood of credit default — using either a classical backend or a simulated quantum circuit (QBN) built with PennyLane.

---

## Features

- Classical model (groupby + empirical joint distributions)
- Quantum backend with real qubit rotation-based inference
- Visual delta plots from simulated interventions
- Built with Streamlit, PennyLane, XGBoost, and Pandas

---

## 📁 Dataset

Uses the [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
Key features used:
- `LIMIT_BAL` – credit limit (proxy for income)
- `Age`       – age group
- `PAY_AMT1`  – recent repayment
- `Default`   – target variable

---

## 💻 Running Locally

```bash
git clone https://github.com/escelctron/qbn_app.git
cd qbn_app
pip install -r requirements.txt
streamlit run qbn_dashboard.py
```


## 🧑‍💻 About the Developer

Developed by **Pranav Sanghadia** — exploring the intersection of Quantum Computing and Explainable AI to bring transparency to financial, healthcare and other domain's decision systems.

🔗 [Connect on LinkedIn](https://www.linkedin.com/in/sanghadia)

---

## 📝 License

MIT License
```
---
