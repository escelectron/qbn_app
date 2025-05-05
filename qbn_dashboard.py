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
from classical_bn import bayesian_inference, bayesian_intervention
from itertools import product

st.set_page_config(page_title="Quantum Explainable AI Dashboard", layout="wide")
st.title("Quantum Bayesian Network: Credit Risk Simulator")

st.markdown("""
This dashboard lets you simulate **credit risk default probability** using interventional inference.
You can compare three backends: 1) Classical 2) Bayesian and 3) Quantum. Select each one from left menu
""")

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

st.sidebar.header("Intervention Controls")
backend = st.sidebar.radio("Select Backend", ["Classical", "Bayesian", "Quantum"], index=2)
limit_bal = st.sidebar.radio("LIMIT_BAL (Income Proxy)", [None, 0, 1], index=0, format_func=lambda x: "No Intervention" if x is None else f"{x} (Low/High)")
age = st.sidebar.radio("Age", [None, 0, 1], index=0, format_func=lambda x: "No Intervention" if x is None else f"{x} (Young/Old)")
pay_amt1 = st.sidebar.radio("PAY_AMT1 (Last Repayment Amount)", [None, 0, 1], index=0, format_func=lambda x: "No Intervention" if x is None else f"{x} (Low/High)")
education = st.sidebar.radio("EDUCATION", [None, 0, 1], index=0, format_func=lambda x: "No Intervention" if x is None else f"{x} (Low/High)")
marriage = st.sidebar.radio("MARRIAGE", [None, 0, 1], index=0, format_func=lambda x: "No Intervention" if x is None else f"{x} (Low/High)")

if backend == "Classical":
    st.info("""
    Classical Method: Empirical probability computed from the dataset.
    For each feature combination, it groups records and calculates the mean of the Default column:

    P(Default=1 | features) ≈ mean(Default)
    """)
elif backend == "Bayesian":
    st.info("""
    Bayesian Method: Probabilistic model built using Classical Bayesian Network.
    The model structure includes: LIMIT_BAL, Age, PAY_AMT1, EDUCATION, MARRIAGE → Default.
    """)
elif backend == "Quantum":
    st.info("""
    Quantum Method: Simulates a quantum circuit using qubit rotations (RY).
    Feature values are encoded as rotation angles, then entangled with a qubit representing Default.

    P(Default=1) is estimated by measuring the qubit amplitude.
    """)

interventions = {}
if limit_bal is not None: interventions['LIMIT_BAL'] = limit_bal
if age is not None: interventions['Age'] = age
if pay_amt1 is not None: interventions['PAY_AMT1'] = pay_amt1
if education is not None: interventions['EDUCATION'] = education
if marriage is not None: interventions['MARRIAGE'] = marriage

feature_columns = ["LIMIT_BAL", "Age", "PAY_AMT1", "EDUCATION", "MARRIAGE"]
profiles = [list(p) for p in product([0, 1], repeat=len(feature_columns))]

@st.cache_data
def make_bar_chart(df, x_col, y_col, title):
    df_sorted = df.copy()
    df_sorted[x_col] = pd.Categorical(df_sorted[x_col], categories=sorted(df_sorted[x_col].unique()), ordered=True)
    #fig = px.bar(df_sorted, x=x_col, y=y_col, title=title, height=400)
    fig = px.bar(df_sorted, x=x_col, y=y_col, title=title, height=400, hover_data=df_sorted.columns)

    fig.update_layout(xaxis_tickangle=-45, margin=dict(l=10, r=10, t=40, b=120))
    return fig

if interventions:
    st.subheader(f"Intervention Results: do({interventions}) using {backend} backend")
    mapped = {k: v for k, v in interventions.items()}

    if backend == "Quantum":
        result_df = pd.DataFrame(run_quantum_inference(profiles, mapped))
    elif backend == "Bayesian":
        result_df = pd.DataFrame(run_bayesian_intervention(profiles, mapped))
    else:
        original = compute_joint_distribution(df)
        intervened = simulate_intervention(df, interventions)
        merged = original.merge(intervened, on=feature_columns, suffixes=("_original", "_do"))
        merged['Delta'] = merged["P(Default=1)_do"] - merged["P(Default=1)_original"]
        merged['P(Default=1) observed'] = merged["P(Default=1)_original"]
        merged['P(Default=1) do()'] = merged["P(Default=1)_do"]
        result_df = merged[feature_columns + ["P(Default=1) observed", "P(Default=1) do()", "Delta"]]

    x_labels = result_df[feature_columns].astype(str).agg('-'.join, axis=1) + "_" + result_df.index.astype(str)
    result_df = result_df.assign(Profile=x_labels)
    #result_df = result_df.sort_values(by="Delta", ascending=False)
    result_df = result_df.sort_values(by="Profile")

    st.plotly_chart(make_bar_chart(result_df, "Profile", "Delta", f"Δ P(Default=1) After {backend} Intervention"), use_container_width=True)
    st.dataframe(result_df)

else:
    st.subheader(f"Default Inference: P(Default=1) without intervention using {backend} backend")
    if backend == "Quantum":
        result_df = pd.DataFrame(run_quantum_inference(profiles, None))
    elif backend == "Bayesian":
        result_df = pd.DataFrame(run_bayesian_inference(profiles))
        result_df.rename(columns={"P(Default=1)": "P(Default=1) observed"}, inplace=True)
    else:
        result_df = compute_joint_distribution(df)
        result_df.rename(columns={"P(Default=1)": "P(Default=1) observed"}, inplace=True)

    x_labels = result_df[feature_columns].astype(str).agg('-'.join, axis=1) + "_" + result_df.index.astype(str)
    result_df = result_df.assign(Profile=x_labels)
    st.plotly_chart(make_bar_chart(result_df, "Profile", "P(Default=1) observed", f"P(Default=1) by Profile - {backend} Backend"), use_container_width=True)
    st.dataframe(result_df)
