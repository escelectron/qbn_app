import pennylane as qml
from pennylane import numpy as np

# 4 qubits: 3 inputs (LIMIT_BAL, Age, PAY_AMT1) + 1 target (Default)
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qbn_circuit(theta, interventions=None):
    """
    Quantum Bayesian Network circuit.
    theta: List of 3 angles for input qubits.
    interventions: Optional dict {index: overridden_angle} to simulate do(X=x)
    """
    for i in range(3):
        angle = theta[i] if interventions is None or i not in interventions else interventions[i]
        qml.RY(angle, wires=i)
        qml.CNOT(wires=[i, 3])  # Entangle with Default qubit

    return qml.probs(wires=3)  # Return probability for Default=0 and Default=1

def angle_map(binary_values):
    """
    Maps binary inputs [0, 1] to RY angles [0.1*pi, 0.9*pi]
    """
    return [0.1 * np.pi if x == 0 else 0.9 * np.pi for x in binary_values]

def run_inference(feature_values, do_intervene=False, intervention_target=None):
    """
    Runs inference on a given binary feature input.
    do_intervene: If True, override specified index with fixed angle.
    intervention_target: dict {feature_index: forced_value}, e.g. {2: 1}
    """
    theta = angle_map(feature_values)

    if do_intervene and intervention_target:
        interventions = {i: angle_map([v])[0] for i, v in intervention_target.items()}
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
            "LIMIT_BAL": profile[0],
            "Age": profile[1],
            "PAY_AMT1": profile[2],
            "P(Default=1) observed": round(p_obs, 4),
            "P(Default=1) do()": round(p_do, 4),
            "Delta": round(p_do - p_obs, 4)
        })
    return results
