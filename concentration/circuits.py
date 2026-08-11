"""ZZ feature-map definitions used by Experiment 4."""

import numpy as np
import pennylane as qml


CIRCUITS = ("no_entanglement", "zz_linear", "zz_full")


def _zz_single_qubit_layer(x, n_qubits):
    d = len(x)

    for wire in range(n_qubits):
        qml.Hadamard(wires=wire)

    for wire in range(n_qubits):
        qml.RZ(2.0 * x[wire % d], wires=wire)


def _zz_linear_entanglement(x, n_qubits):
    d = len(x)

    for wire in range(n_qubits - 1):
        angle = 2.0 * (np.pi - x[wire % d]) * (
            np.pi - x[(wire + 1) % d]
        )
        qml.CNOT(wires=[wire, wire + 1])
        qml.RZ(angle, wires=wire + 1)
        qml.CNOT(wires=[wire, wire + 1])


def _zz_full_entanglement(x, n_qubits):
    d = len(x)

    for control in range(n_qubits):
        for target in range(control + 1, n_qubits):
            angle = 2.0 * (np.pi - x[control % d]) * (
                np.pi - x[target % d]
            )
            qml.CNOT(wires=[control, target])
            qml.RZ(angle, wires=target)
            qml.CNOT(wires=[control, target])


def build_feature_map(circuit_name, n_qubits, n_layers):
    """Return a callable implementing the requested ZZ feature map."""
    if circuit_name not in CIRCUITS:
        raise ValueError(f"Unknown circuit: {circuit_name}")

    def feature_map(x):
        for _ in range(n_layers):
            _zz_single_qubit_layer(x, n_qubits)

            if circuit_name == "zz_linear":
                _zz_linear_entanglement(x, n_qubits)
            elif circuit_name == "zz_full":
                _zz_full_entanglement(x, n_qubits)

    return feature_map
