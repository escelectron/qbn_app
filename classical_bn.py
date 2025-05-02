"""
Quantum Explainable AI Dashboard
Author: Pranav Sanghadia
LinkedIn: https://www.linkedin.com/in/sanghadia
License: MIT
"""
# File Name: classical_bn.py

import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from interventional_V1_0 import load_data

def build_model():
    df = load_data()
    model = BayesianNetwork([
        ("LIMIT_BAL", "Default"),
        ("Age", "Default"),
        ("PAY_AMT1", "Default")
    ])
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    return model

# Build once and reuse
_bayesian_model = build_model()
_infer = VariableElimination(_bayesian_model)

# Mapping for index to column name
features = ["LIMIT_BAL", "Age", "PAY_AMT1"]

def bayesian_inference(profiles):
    results = []
    for p in profiles:
        evidence = dict(zip(features, p))
        q = _infer.query(variables=["Default"], evidence=evidence, show_progress=False)
        prob = round(q.values[1], 4)
        results.append({**evidence, "P(Default=1) observed": prob})
    return results

def bayesian_intervention(profiles, intervention_target):
    results = []
    for p in profiles:
        # Override the features using intervention
        intervened = p.copy()
        for idx, val in intervention_target.items():
            intervened[idx] = val
        evidence = dict(zip(features, intervened))
        q = _infer.query(variables=["Default"], evidence=evidence, show_progress=False)
        p_do = round(q.values[1], 4)

        # Also compute the original probability
        orig_evidence = dict(zip(features, p))
        q_orig = _infer.query(variables=["Default"], evidence=orig_evidence, show_progress=False)
        p_obs = round(q_orig.values[1], 4)

        results.append({
            "LIMIT_BAL": p[0],
            "Age": p[1],
            "PAY_AMT1": p[2],
            "P(Default=1) observed": p_obs,
            "P(Default=1) do()": p_do,
            "Delta": round(p_do - p_obs, 4)
        })
    return results
