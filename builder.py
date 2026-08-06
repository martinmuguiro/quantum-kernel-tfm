import pennylane as qml

from layer import create_layer


def build_ansatz(x, parameters, config, n_qubits, n_layers):
    """
    Build the parameterized quantum ansatz.
    """

    if parameters is not None and len(parameters) != n_layers:
        raise ValueError(
            f"params_layers must have length {n_layers}, "
            f"got {len(parameters)}."
        )

    for wire in range(n_qubits):
        qml.RY(x[wire % 2], wires=wire)

    for layer_idx in range(n_layers):
        layer_params = None if parameters is None else parameters[layer_idx]

        create_layer(
            config,
            n_qubits=n_qubits,
            params=layer_params,
        )

        for wire in range(n_qubits):
            qml.RY(x[wire % 2], wires=wire)
