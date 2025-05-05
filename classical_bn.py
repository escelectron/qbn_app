"""
Quantum Explainable AI Dashboard
Author: Pranav Sanghadia
LinkedIn: https://www.linkedin.com/in/sanghadia
License: MIT
"""
# File Name: classical_bn.py

from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from interventional_V1_0 import load_data

feature_columns = ["LIMIT_BAL", "Age", "PAY_AMT1", "EDUCATION", "MARRIAGE"]

def build_model():
    df = load_data()
    model = BayesianNetwork([(feat, "Default") for feat in feature_columns])
    df[feature_columns] = df[feature_columns].astype(int)
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    return model

_bayesian_model = build_model()
_infer = VariableElimination(_bayesian_model)

def bayesian_inference(profiles):
    results = []
    for p in profiles:
        evidence = dict(zip(feature_columns, p))
        q = _infer.query(variables=["Default"], evidence=evidence, show_progress=False)
        prob = round(q.values[1], 4)
        results.append({**evidence, "P(Default=1)": prob})
    return results

def bayesian_intervention(profiles, intervention_target):
    results = []
    feature_index = {name: i for i, name in enumerate(feature_columns)}
    for p in profiles:
        intervened = p.copy()
        for name, val in intervention_target.items():
            if name in feature_index:
                intervened[feature_index[name]] = val
        evidence = dict(zip(feature_columns, intervened))
        q = _infer.query(variables=["Default"], evidence=evidence, show_progress=False)
        p_do = round(q.values[1], 4)

        orig_evidence = dict(zip(feature_columns, p))
        q_orig = _infer.query(variables=["Default"], evidence=orig_evidence, show_progress=False)
        p_obs = round(q_orig.values[1], 4)

        results.append({
            **orig_evidence,
            "P(Default=1) observed": p_obs,
            "P(Default=1) do()": p_do,
            "Delta": round(p_do - p_obs, 4)
        })
    return results