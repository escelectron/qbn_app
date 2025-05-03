"""
Quantum Explainable AI Dashboard
Author: Pranav Sanghadia
LinkedIn: https://www.linkedin.com/in/sanghadia
License: MIT
"""
# File Name: qbn_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
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
@st.cache_data
def get_data():
    return load_data()

@st.cache_data
def run_quantum_inference(profiles, intervention_target):
    return batch_inference(profiles, intervention_target)

@st.cache_data
def run_bayesian_inference(profiles):
    return bayesian_inference(profiles)

@st.cache_data
def run_bayesian_intervention(profiles, mapped):
    return bayesian_intervention(profiles, mapped)

df = get_data()

st.sidebar.header("🔧 Intervention Controls")
backend = st.sidebar.radio("Select Backend", ["Classical", "Bayesian", "Quantum"], index=2)
limit_bal = st.sidebar.radio("LIMIT_BAL (Income Proxy)", [None, 0, 1], index=0, format_func=lambda x: "No Intervention" if x is None else f"{x} (Low/High)")
age = st.sidebar.radio("Age", [None, 0, 1], index=0, format_func=lambda x: "No Intervention" if x is None else f"{x} (Young/Old)")
pay_amt1 = st.sidebar.radio("PAY_AMT1 (Last Repayment Amount)", [None, 0, 1], index=0, format_func=lambda x: "No Intervention" if x is None else f"{x} (Low/High)")

# Show method explanation
if backend == "Classical":
    st.info("""
    🏠 **Classical Method**: Empirical probability computed from the dataset.
    For each feature combination, it groups records and calculates the mean of the Default column:

    **P(Default=1 | features) ≈ mean(Default)**

    This reflects the fraction of historical defaults in the dataset for that feature group.
    """)
elif backend == "Bayesian":
    st.info("""
    🧪 **Bayesian Method**: Probabilistic model built using Classical Bayesian Network.
    It defines conditional dependencies between features and computes probability using inference:

    **P(Default=1 | features)** and **P(Default=1 | do(X = x))**

    The model structure is: LIMIT_BAL, Age, PAY_AMT1 → Default.
    """)
elif backend == "Quantum":
    st.info("""
    Ψ **Quantum Method**: Simulates a quantum circuit using qubit rotations (RY).
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

@st.cache_data
def make_bar_chart(df, x_col, y_col, title):
    fig = px.bar(df, x=x_col, y=y_col, title=title, height=400)
    return fig

if interventions:
    st.subheader(f"📈 Intervention Results: do({interventions}) using {backend} backend")

    feature_index = {"LIMIT_BAL": 0, "Age": 1, "PAY_AMT1": 2}
    mapped = {feature_index[k]: v for k, v in interventions.items()}

    if backend == "Quantum":
        result_df = pd.DataFrame(run_quantum_inference(profiles, mapped))
    elif backend == "Bayesian":
        result_df = pd.DataFrame(run_bayesian_intervention(profiles, mapped))
    else:
        original = compute_joint_distribution(df)
        intervened = simulate_intervention(df, interventions)
        merged = original.merge(intervened, on=["LIMIT_BAL", "Age", "PAY_AMT1"], suffixes=("_original", "_do"))
        merged['Delta'] = merged["P(Default=1)_do"] - merged["P(Default=1)_original"]
        merged['P(Default=1) observed'] = merged["P(Default=1)_original"]
        merged['P(Default=1) do()'] = merged["P(Default=1)_do"]
        result_df = merged[["LIMIT_BAL", "Age", "PAY_AMT1", "P(Default=1) observed", "P(Default=1) do()", "Delta"]]

    x_labels = result_df[["LIMIT_BAL", "Age", "PAY_AMT1"]].astype(str).agg('-'.join, axis=1)
    st.plotly_chart(make_bar_chart(result_df.assign(Profile=x_labels), "Profile", "Delta" if interventions else "P(Default=1) observed", f"Δ P(Default=1) After {backend} Intervention" if interventions else f"P(Default=1) by Profile - {backend} Backend"), use_container_width=True)

    st.markdown("""
    The table below shows how the probability of default changes for each feature combination after applying the selected intervention.
    A positive delta means the intervention increased the risk of default, while a negative delta means it reduced the risk.
    """)
    st.dataframe(result_df)

else:
    st.subheader(f"📊 Default Inference: P(Default=1) without intervention using {backend} backend")
    if backend == "Quantum":
        result_df = pd.DataFrame(run_quantum_inference(profiles, None))
    elif backend == "Bayesian":
        result_df = pd.DataFrame(run_bayesian_inference(profiles))
    else:
        result_df = compute_joint_distribution(df)
        result_df.rename(columns={"P(Default=1)": "P(Default=1) observed"}, inplace=True)

    x_labels = result_df[["LIMIT_BAL", "Age", "PAY_AMT1"]].astype(str).agg('-'.join, axis=1)
    st.plotly_chart(make_bar_chart(result_df.assign(Profile=x_labels), "Profile", "P(Default=1) observed", f"P(Default=1) by Profile - {backend} Backend"), use_container_width=True)

    st.markdown("""
    This plot shows the default risk for each combination of features without applying any intervention. These are the baseline probabilities based on the selected backend.
    """)
    st.dataframe(result_df)
