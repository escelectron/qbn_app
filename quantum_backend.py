"""
Quantum Explainable AI Backend
Author: Pranav Sanghadia
LinkedIn: https://www.linkedin.com/in/sanghadia
License: MIT
"""

import pennylane as qml
from pennylane import numpy as np

feature_columns = ["LIMIT_BAL", "Age", "PAY_AMT1", "EDUCATION", "MARRIAGE"]
n_qubits = len(feature_columns) + 1  # +1 for the Default qubit
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qbn_circuit(theta, interventions=None):
    """
    Quantum Bayesian Network circuit with parameterized entanglement.
    theta: List of angles for input qubits.
    interventions: Optional dict {index: overridden_angle} to simulate do(X=x)
    """
    assert len(theta) == n_qubits - 1, f"Expected {n_qubits - 1} input angles."

    for i in range(n_qubits - 1):
        angle = theta[i] if interventions is None or i not in interventions else interventions[i]
        qml.RY(angle, wires=i)
        qml.CRY(angle, wires=[i, n_qubits - 1])  # Parameterized entanglement

    return qml.probs(wires=n_qubits - 1)  # Measure the Default qubit

def angle_map(binary_values):
    """
    Maps binary inputs [0, 1] to RY angles [0.1*pi, 0.9*pi]
    """
    return [0.1 * np.pi if x == 0 else 0.9 * np.pi for x in binary_values]

def run_inference(feature_values, do_intervene=False, intervention_target=None):
    """
    Runs inference on a given binary feature input.
    do_intervene: If True, override specified feature name with fixed angle.
    intervention_target: dict {feature_name: forced_value}, e.g. {"PAY_AMT1": 1}
    """
    theta = angle_map(feature_values)

    if do_intervene and intervention_target:
        feature_index = {name: idx for idx, name in enumerate(feature_columns)}
        interventions = {feature_index[k]: angle_map([v])[0] for k, v in intervention_target.items() if k in feature_index}
        probs = qbn_circuit(theta, interventions)
    else:
        probs = qbn_circuit(theta)

    return float(probs[1])  # Return P(Default=1)

def batch_inference(profiles, intervention_target=None):
    """
    Run batch inference for a list of feature combinations.
    """
    results = []
    for profile in profiles:
        p_obs = run_inference(profile, do_intervene=False)
        p_do = run_inference(profile, do_intervene=True, intervention_target=intervention_target)
        results.append({
            **{feature_columns[i]: profile[i] for i in range(len(feature_columns))},
            "P(Default=1) observed": round(p_obs, 4),
            "P(Default=1) do()": round(p_do, 4),
            "Delta": round(p_do - p_obs, 4)
        })
    return results
