import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from interventional_V1_0 import load_data, compute_joint_distribution, simulate_intervention
from quantum_backend import batch_inference

st.set_page_config(page_title="Quantum Explainable AI Dashboard", layout="wide")
st.title("🔮 Quantum Bayesian Network: Credit Risk Simulator")

st.markdown("""
This dashboard lets you simulate **credit risk default probability** using interventional inference.
You can compare two backends:
- 🧠 **Classical**: Computes empirical probabilities via groupby aggregation.
- 🧪 **Quantum**: Simulates a quantum circuit where feature values are encoded as qubit rotations.

Use the sidebar to intervene (force) one or more feature values and observe the change in risk.
""")

# Load and show dataset
df = load_data()

st.sidebar.header("🔧 Intervention Controls")
backend = st.sidebar.radio("Select Backend", ["Classical", "Quantum"], index=1)
limit_bal = st.sidebar.radio("LIMIT_BAL (Income Proxy)", [None, 0, 1], index=0, format_func=lambda x: "No Intervention" if x is None else f"{x} (Low/High)")
age = st.sidebar.radio("Age", [None, 0, 1], index=0, format_func=lambda x: "No Intervention" if x is None else f"{x} (Young/Old)")
pay_amt1 = st.sidebar.radio("PAY_AMT1 (Last Repayment Amount)", [None, 0, 1], index=0, format_func=lambda x: "No Intervention" if x is None else f"{x} (Low/High)")

interventions = {}
if limit_bal is not None: interventions['LIMIT_BAL'] = limit_bal
if age is not None: interventions['Age'] = age
if pay_amt1 is not None: interventions['PAY_AMT1'] = pay_amt1

# All 8 binary feature combinations
profiles = [
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
]

if interventions:
    st.subheader(f"📈 Intervention Results: do({interventions}) using {backend} backend")

    feature_index = {"LIMIT_BAL": 0, "Age": 1, "PAY_AMT1": 2}
    mapped = {feature_index[k]: v for k, v in interventions.items()}

    if backend == "Quantum":
        result_df = pd.DataFrame(batch_inference(profiles, intervention_target=mapped))
    else:
        original = compute_joint_distribution(df)
        intervened = simulate_intervention(df, interventions)
        merged = original.merge(intervened, on=["LIMIT_BAL", "Age", "PAY_AMT1"], suffixes=("_original", "_do"))
        merged['Delta'] = merged["P(Default=1)_do"] - merged["P(Default=1)_original"]
        merged['P(Default=1) observed'] = merged["P(Default=1)_original"]
        merged['P(Default=1) do()'] = merged["P(Default=1)_do"]
        result_df = merged[["LIMIT_BAL", "Age", "PAY_AMT1", "P(Default=1) observed", "P(Default=1) do()", "Delta"]]

    st.dataframe(result_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=result_df, x=result_df[["LIMIT_BAL", "Age", "PAY_AMT1"]].astype(str).agg('-'.join, axis=1), y='Delta', ax=ax)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title(f"Δ P(Default=1) After {backend} Intervention")
    ax.set_ylabel("Change in P(Default=1)")
    ax.set_xlabel("Feature Combination (LIMIT_BAL-Age-PAY_AMT1)")
    st.pyplot(fig)
else:
    st.info("Use the sidebar to select an intervention and run a simulation.")
