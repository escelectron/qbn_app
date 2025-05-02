"""
Quantum Explainable AI Dashboard
Author: Pranav Sanghadia
LinkedIn: https://www.linkedin.com/in/sanghadia
License: MIT
"""
# File Name: qbn_dashboard.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from interventional_V1_0 import load_data, compute_joint_distribution, simulate_intervention
from quantum_backend import batch_inference
from classical_bn import bayesian_inference, bayesian_intervention  # NEW IMPORT

st.set_page_config(page_title="Quantum Explainable AI Dashboard", layout="wide")
st.title("⚛️ Quantum Bayesian Network: Credit Risk Simulator")

st.markdown("""
This dashboard lets you simulate **credit risk default probability** using interventional inference.
You can compare three backends: 1) Classical 2) Bayesian and 3) Quantum. Select each one from left menu
""")

# Load and show dataset
df = load_data()

st.sidebar.header("🔧 Intervention Controls")
backend = st.sidebar.radio("Select Backend", ["Classical", "Bayesian", "Quantum"], index=2)
limit_bal = st.sidebar.radio("LIMIT_BAL (Income Proxy)", [None, 0, 1], index=0, format_func=lambda x: "No Intervention" if x is None else f"{x} (Low/High)")
age = st.sidebar.radio("Age", [None, 0, 1], index=0, format_func=lambda x: "No Intervention" if x is None else f"{x} (Young/Old)")
pay_amt1 = st.sidebar.radio("PAY_AMT1 (Last Repayment Amount)", [None, 0, 1], index=0, format_func=lambda x: "No Intervention" if x is None else f"{x} (Low/High)")

# Show method explanation
if backend == "Classical":
    st.info("""
    🧠 **Classical Method**: Empirical probability computed from the dataset.
    For each feature combination, it groups records and calculates the mean of the Default column:

    **P(Default=1 | features) ≈ mean(Default)**

    This reflects the fraction of historical defaults in the dataset for that feature group.
    """)
elif backend == "Bayesian":
    st.info("""
    📊 **Bayesian Method**: Probabilistic model built using Classical Bayesian Network.
    It defines conditional dependencies between features and computes probability using inference:

    **P(Default=1 | features)** and **P(Default=1 | do(X = x))**

    The model structure is: LIMIT_BAL, Age, PAY_AMT1 → Default.
    """)
elif backend == "Quantum":
    st.info("""
    🧪 **Quantum Method**: Simulates a quantum circuit using qubit rotations (RY).
    Feature values are encoded as rotation angles, then entangled with a qubit representing Default.

    **P(Default=1)** is estimated by measuring the qubit amplitude.

    Interventions override rotation angles directly, simulating **do(X = x)**.
    """)

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
    elif backend == "Bayesian":
        result_df = pd.DataFrame(bayesian_intervention(profiles, mapped))
    else:
        original = compute_joint_distribution(df)
        intervened = simulate_intervention(df, interventions)
        merged = original.merge(intervened, on=["LIMIT_BAL", "Age", "PAY_AMT1"], suffixes=("_original", "_do"))
        merged['Delta'] = merged["P(Default=1)_do"] - merged["P(Default=1)_original"]
        merged['P(Default=1) observed'] = merged["P(Default=1)_original"]
        merged['P(Default=1) do()'] = merged["P(Default=1)_do"]
        result_df = merged[["LIMIT_BAL", "Age", "PAY_AMT1", "P(Default=1) observed", "P(Default=1) do()", "Delta"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=result_df, x=result_df[["LIMIT_BAL", "Age", "PAY_AMT1"]].astype(str).agg('-'.join, axis=1), y='Delta', ax=ax)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title(f"Δ P(Default=1) After {backend} Intervention")
    ax.set_ylabel("Change in P(Default=1)")
    ax.set_xlabel("Feature Combination (LIMIT_BAL-Age-PAY_AMT1)")
    st.pyplot(fig)

    st.markdown("""
    The table below shows how the probability of default changes for each feature combination after applying the selected intervention.
    A positive delta means the intervention increased the risk of default, while a negative delta means it reduced the risk.
    """)
    st.dataframe(result_df)

else:
    st.subheader(f"📊 Default Inference: P(Default=1) without intervention using {backend} backend")
    if backend == "Quantum":
        result_df = pd.DataFrame(batch_inference(profiles, intervention_target=None))
    elif backend == "Bayesian":
        result_df = pd.DataFrame(bayesian_inference(profiles))
    else:
        result_df = compute_joint_distribution(df)
        result_df.rename(columns={"P(Default=1)": "P(Default=1) observed"}, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=result_df, x=result_df[["LIMIT_BAL", "Age", "PAY_AMT1"]].astype(str).agg('-'.join, axis=1), y='P(Default=1) observed', ax=ax)
    ax.set_title(f"P(Default=1) by Profile - {backend} Backend")
    ax.set_ylabel("P(Default=1)")
    ax.set_xlabel("Feature Combination (LIMIT_BAL-Age-PAY_AMT1)")
    st.pyplot(fig)

    st.markdown("""
    This plot shows the default risk for each combination of features without applying any intervention. These are the baseline probabilities based on the selected backend.
    """)
    st.dataframe(result_df)

