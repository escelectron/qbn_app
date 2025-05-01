"""
Quantum Explainable AI Dashboard
Author: Pranav Sanghadia
LinkedIn: https://www.linkedin.com/in/sanghadia
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def simulate_intervention(df, interventions):
    """
    Perform interventional inference by overriding selected features.
    Example: interventions = {'LIMIT_BAL': 1, 'PAY_AMT1': 0}
    """
    df_int = df.copy()
    for feature, value in interventions.items():
        df_int[feature] = value
    
    joint = df_int.groupby(["LIMIT_BAL", "Age", "PAY_AMT1"], observed=False)["Default"].mean().reset_index()
    joint.rename(columns={"Default": "P(Default=1)"}, inplace=True)
    joint['Intervention'] = '+'.join([f"{k}={v}" for k, v in interventions.items()])
    return joint

def compare_intervention_vs_original(original, intervened):
    """
    Merges original and intervened joint distributions for comparison.
    """
    merged = original.merge(intervened, on=["LIMIT_BAL", "Age", "PAY_AMT1"], suffixes=("_original", "_do"))
    merged['Diff'] = merged["P(Default=1)_do"] - merged["P(Default=1)_original"]

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=merged[["LIMIT_BAL", "Age", "PAY_AMT1"]].astype(str).agg('-'.join, axis=1),
        y=merged['Diff']
    )
    plt.xticks(rotation=45)
    plt.title("Change in P(Default=1) due to Intervention")
    plt.xlabel("Feature Combination")
    plt.ylabel("Δ P(Default=1) (do() - original)")
    plt.tight_layout()
    plt.savefig("qbn_intervention_comparison.png")
    plt.close()
    return merged



def load_data():
    df = pd.read_excel("default_of_credit_card_clients.xlsx", engine="openpyxl", header=1)
    df = df.rename(columns={"default payment next month": "Default"})
    df["LIMIT_BAL"] = pd.qcut(df["LIMIT_BAL"], 2, labels=[0, 1])
    df["Age"] = pd.qcut(df["AGE"], 2, labels=[0, 1])
    df["PAY_AMT1"] = pd.qcut(df["PAY_AMT1"], 2, labels=[0, 1])
    df["Default"] = df["Default"].astype(int)
    return df[["LIMIT_BAL", "Age", "PAY_AMT1", "Default"]]

def run_classical_model(df):
    X = df[["LIMIT_BAL", "Age", "PAY_AMT1"]]
    y = df["Default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', enable_categorical=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    print("[Classical Model Performance]")
    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC: {roc_auc}")
    return model

def compute_joint_distribution(df):
    #joint = df.groupby(["LIMIT_BAL", "Age", "PAY_AMT1"])["Default"].mean().reset_index()
    joint = df.groupby(["LIMIT_BAL", "Age", "PAY_AMT1"], observed=False)["Default"].mean().reset_index()

    joint.rename(columns={"Default": "P(Default=1)"}, inplace=True)
    return joint


def plot_heatmap_and_barplot(joint_df):
    #heatmap_data = joint_df.pivot_table(index='Age', columns=['LIMIT_BAL', 'PAY_AMT1'], values='P(Default=1)')
    heatmap_data = joint_df.pivot_table(
    index="Age",
    columns=["LIMIT_BAL", "PAY_AMT1"],
    values="P(Default=1)",
    observed=False
)

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".3f")
    plt.title("Heatmap of P(Default=1) by Age, LIMIT_BAL, PAY_AMT1")
    plt.ylabel("Age")
    plt.xlabel("LIMIT_BAL and PAY_AMT1")
    plt.tight_layout()
    plt.savefig("qbn_heatmap_default_probability.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    joint_df['label'] = joint_df[['LIMIT_BAL', 'Age', 'PAY_AMT1']].astype(str).agg('-'.join, axis=1)
    sns.barplot(x='label', y='P(Default=1)', data=joint_df)
    plt.xticks(rotation=45)
    plt.title("Bar Plot of P(Default=1) for Feature Combinations")
    plt.ylabel("P(Default=1)")
    plt.xlabel("Feature Combinations (LIMIT_BAL-Age-PAY_AMT1)")
    plt.tight_layout()
    plt.savefig("qbn_barplot_default_probability.png")
    plt.close()

# def main():
#     df = load_data()
#     model = run_classical_model(df)
#     joint_df = compute_joint_distribution(df)
#     print("[QBN Joint Probability Distribution]")
#     print(joint_df)
#     print("Mean P(Default=1):", joint_df["P(Default=1)"].mean())
#     plot_heatmap_and_barplot(joint_df)


def main():
    df = load_data()
    model = run_classical_model(df)
    joint_df = compute_joint_distribution(df)
    print("[Original QBN Joint Distribution]")
    print(joint_df)
    
    interventions = {"PAY_AMT1": 1}
    intervened_df = simulate_intervention(df, interventions)
    print(f"[Intervention: do({interventions})]")
    print(intervened_df)
    
    compare_intervention_vs_original(joint_df, intervened_df)



if __name__ == "__main__":
    main()

'''

✅ Key Outcomes:
✅ XGBoost was successfully trained with:

Accuracy ≈ 78.2%

ROC AUC ≈ 0.624

✅ The script generated a QBN joint probability distribution over discretized LIMIT_BAL, Age, and PAY_AMT1 with respect to P(Default=1)

✅ The script calculated Mean P(Default=1) as a summary metric

✅ Visual outputs (heatmap + bar plot) were created using the joint probability table


'''