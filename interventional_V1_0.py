"""
Quantum Explainable AI Dashboard
Author: Pranav Sanghadia
LinkedIn: https://www.linkedin.com/in/sanghadia
License: MIT
"""

import pandas as pd

feature_columns = ["LIMIT_BAL", "Age", "PAY_AMT1", "EDUCATION", "MARRIAGE"]

def load_data():
    df = pd.read_excel("default_of_credit_card_clients.xlsx", engine="openpyxl", header=1)
    df = df.rename(columns={"default payment next month": "Default"})
    df["LIMIT_BAL"] = pd.qcut(df["LIMIT_BAL"], 2, labels=[0, 1])
    df["Age"] = pd.qcut(df["AGE"], 2, labels=[0, 1])
    df["PAY_AMT1"] = pd.qcut(df["PAY_AMT1"], 2, labels=[0, 1])
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 1 if x >= 2 else 0)
    df["MARRIAGE"] = df["MARRIAGE"].apply(lambda x: 1 if x == 1 else 0)
    df["Default"] = df["Default"].astype(int)
    return df[feature_columns + ["Default"]]

def compute_joint_distribution(df):
    joint = df.groupby(feature_columns, observed=False)["Default"].mean().reset_index()
    joint.rename(columns={"Default": "P(Default=1)"}, inplace=True)
    return joint

def simulate_intervention(df, interventions):
    df_int = df.copy()
    for feature, value in interventions.items():
        df_int[feature] = value
    joint = df_int.groupby(feature_columns, observed=False)["Default"].mean().reset_index()
    joint.rename(columns={"Default": "P(Default=1)"}, inplace=True)
    joint["Intervention"] = "+".join([f"{k}={v}" for k, v in interventions.items()])
    return joint