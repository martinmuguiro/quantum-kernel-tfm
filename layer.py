import pennylane as qml


def apply_hadamards(n_qubits):
    """Apply a Hadamard gate to every qubit."""
    for wire in range(n_qubits):
        qml.Hadamard(wires=wire)


def apply_single_qubit_rotations(params, rotation_type="ry", target=None):
    """
    Apply single-qubit rotation gates.

    Args:
        params: Rotation parameters.
        rotation_type: One of {"rx", "ry", "rz"}.
        target: Target qubit. If None, apply to all qubits.
    """

    rotations = {
        "rx": qml.RX,
        "ry": qml.RY,
        "rz": qml.RZ,
    }

    try:
        gate = rotations[rotation_type]
    except KeyError:
        raise ValueError(f"Unknown rotation type: {rotation_type}")

    if target is None:
        for wire, angle in enumerate(params):
            gate(angle, wires=wire)
    else:
        gate(params[target], wires=target)


def apply_entanglement(
    n_qubits,
    ent_type="none",
    layout="NN",
    params=None,
    inverse=False,
    fixed_pair=None,
):
    """
    Apply an entangling layer.
    """

    if ent_type == "none":
        return

    if fixed_pair is not None:
        q1, q2 = fixed_pair

        if ent_type == "cnot":
            qml.CNOT(wires=[q1, q2])
        elif ent_type == "cz":
            qml.CZ(wires=[q1, q2])
        elif ent_type == "crx":
            qml.CRX(params[0], wires=[q1, q2])
        elif ent_type == "crz":
            qml.CRZ(params[0], wires=[q1, q2])

        return

    if layout.lower() == "nn":
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]

    elif layout.lower() == "cb":
        pairs = (
            [(i, i + 1) for i in range(n_qubits - 1)]
            + [(n_qubits - 1, 0)]
        )

    elif layout.lower() == "all":
        pairs = [
            (i, j)
            for i in reversed(range(n_qubits))
            for j in reversed(range(n_qubits))
            if i != j
        ]

    else:
        raise ValueError(f"Unknown entanglement layout: {layout}")

    if inverse:
        pairs = [(q2, q1) for q1, q2 in reversed(pairs)]

    if ent_type == "crx":
        for (q1, q2), angle in zip(pairs, params):
            qml.CRX(angle, wires=[q1, q2])

    elif ent_type == "crz":
        for (q1, q2), angle in zip(pairs, params):
            qml.CRZ(angle, wires=[q1, q2])

    elif ent_type == "cnot":
        for q1, q2 in pairs:
            qml.CNOT(wires=[q1, q2])

    elif ent_type == "cz":
        for q1, q2 in pairs:
            qml.CZ(wires=[q1, q2])


def create_layer(config, n_qubits=4, params=None):
    """
    Build a circuit layer from a configuration string.
    """

    gates = [gate.strip().lower() for gate in config.split(",")]

    for index, gate in enumerate(gates):

        if gate.startswith(("rx", "ry", "rz")):

            parts = gate.split("-")
            rotation = parts[0]
            target = None if len(parts) == 1 else int(parts[1])

            apply_single_qubit_rotations(
                params[index],
                rotation_type=rotation,
                target=target,
            )

        elif any(name in gate for name in ("cnot", "cz", "crx", "crz")):

            parts = gate.split("-")

            entanglement = parts[0]
            layout = None
            inverse = False
            fixed_pair = None

            if len(parts) == 2:
                if parts[1].isdigit():
                    pair = parts[1]

                    if len(pair) != 2:
                        raise ValueError(
                            f"Invalid fixed entanglement pair: {pair}"
                        )

                    fixed_pair = (int(pair[0]), int(pair[1]))
                else:
                    layout = parts[1]

            elif len(parts) == 3:
                layout = parts[1]
                inverse = parts[2] == "inv"

            if fixed_pair is None and layout is None:
                raise ValueError(
                    f"Entanglement layout missing in '{gate}'."
                )

            apply_entanglement(
                n_qubits=n_qubits,
                ent_type=entanglement,
                layout=layout if fixed_pair is None else None,
                params=params[index] if entanglement in ("crx", "crz") else None,
                inverse=inverse,
                fixed_pair=fixed_pair,
            )

        elif gate == "h":
            apply_hadamards(n_qubits)

        elif gate == "barrier":
            qml.Barrier(wires=range(n_qubits))

        else:
            raise ValueError(f"Unknown gate specification: {gate}")
