import numpy as np
import pennylane as qml

from builder import build_ansatz


_SINGLE_PARAMETER_GATES = {"RX", "RY", "RZ"}
_CONTROLLED_PARAMETER_GATES = {"CRX", "CRY", "CRZ"}


def params(config_str, n_qubits, n_layers, low=-np.pi, high=np.pi):
    """
    Generate the parameter structure for a variational ansatz.

    Returns a nested list with the structure:
        [layer][gate][parameters]
    """

    config = [gate.strip().upper() for gate in config_str.split(",")]
    result = []

    for _ in range(n_layers):
        layer_params = []

        for gate in config:
            parts = gate.split("-")
            base_gate = parts[0]
            entanglement = parts[1] if len(parts) > 1 else None

            if base_gate in _SINGLE_PARAMETER_GATES:
                n_params = n_qubits

            elif base_gate in _CONTROLLED_PARAMETER_GATES:
                if entanglement is None:
                    raise ValueError(
                        f"Entanglement layout missing for '{gate}'."
                    )

                entanglement = entanglement.upper()

                if entanglement == "NN":
                    n_params = n_qubits - 1
                elif entanglement == "CB":
                    n_params = n_qubits
                elif entanglement == "ALL":
                    n_params = n_qubits * (n_qubits - 1)
                elif entanglement.isdigit() and len(entanglement) == 2:
                    n_params = 1
                else:
                    raise ValueError(
                        f"Unknown entanglement type in '{gate}'."
                    )

            else:
                layer_params.append(None)
                continue

            layer_params.append(
                list(np.random.uniform(low, high, n_params))
            )

        result.append(layer_params)

    return result


def _build_feature_map(parameters, config, n_qubits, n_layers, gpu):
    """Create the quantum feature map QNode."""

    device_name = "lightning.gpu" if gpu else "lightning.qubit"
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def feature_map(x1, x2):
        build_ansatz(x1, parameters, config, n_qubits, n_layers)
        qml.adjoint(build_ansatz)(
            x2,
            parameters,
            config,
            n_qubits,
            n_layers,
        )
        return qml.probs(wires=range(n_qubits))

    return feature_map


def quantum_kernel(X, config, n_qubits, n_layers, gpu=False):
    """Compute the kernel matrix for a dataset."""

    parameters = params(config, n_qubits, n_layers)
    feature_map = _build_feature_map(
        parameters,
        config,
        n_qubits,
        n_layers,
        gpu,
    )

    n_samples = len(X)
    kernel = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            value = feature_map(X[i], X[j])[0]
            kernel[i, j] = value
            kernel[j, i] = value

    return kernel


def quantum_kernel_cross(X1, X2, config, n_qubits, n_layers, gpu=False):
    """Compute the cross-kernel matrix between two datasets."""

    parameters = params(config, n_qubits, n_layers)
    feature_map = _build_feature_map(
        parameters,
        config,
        n_qubits,
        n_layers,
        gpu,
    )

    kernel = np.zeros((len(X1), len(X2)))

    for i in range(len(X1)):
        for j in range(len(X2)):
            kernel[i, j] = feature_map(X1[i], X2[j])[0]

    return kernel
