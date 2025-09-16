# qnn/observables/custom.py
from __future__ import annotations
from typing import Iterable, Literal, Tuple
import numpy as np
from qiskit.quantum_info import SparsePauliOp

# ---------- Simple local / global Pauli observables ----------

def z_on(qubit: int, num_qubits: int) -> SparsePauliOp:
    """Z acting on a single qubit index; I elsewhere."""
    z = ["I"] * num_qubits
    z[qubit] = "Z"
    return SparsePauliOp("".join(reversed(z)))  # Qiskit uses little-endian strings


def parity_z(num_qubits: int) -> SparsePauliOp:
    """Z ⊗ Z ⊗ ... ⊗ Z (global parity)."""
    return SparsePauliOp("".join(reversed("Z" * num_qubits)))


def x_on(qubit: int, num_qubits: int) -> SparsePauliOp:
    x = ["I"] * num_qubits
    x[qubit] = "X"
    return SparsePauliOp("".join(reversed(x)))


def y_on(qubit: int, num_qubits: int) -> SparsePauliOp:
    y = ["I"] * num_qubits
    y[qubit] = "Y"
    return SparsePauliOp("".join(reversed(y)))


# ---------- Ising and related Hamiltonians ----------

def ising_hamiltonian(
    num_qubits: int,
    J: float = 1.0,
    h: float = 0.0,
    topology: Literal["line", "ring", "full"] = "line",
) -> SparsePauliOp:
    """
    H = -J * sum_{<i,j>} Z_i Z_j  -  h * sum_i X_i
    where neighbor set depends on topology.
    """
    paulis = []
    coeffs = []

    # ZZ couplings
    def add_zz(i, j):
        s = ["I"] * num_qubits
        s[i] = "Z"; s[j] = "Z"
        paulis.append("".join(reversed(s)))
        coeffs.append(-J)

    if topology in {"line", "ring"}:
        for i in range(num_qubits - 1):
            add_zz(i, i + 1)
        if topology == "ring" and num_qubits > 2:
            add_zz(num_qubits - 1, 0)
    elif topology == "full":
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                add_zz(i, j)
    else:
        raise ValueError("topology must be one of {'line','ring','full'}")

    # Transverse field X
    if abs(h) > 0:
        for i in range(num_qubits):
            s = ["I"] * num_qubits
            s[i] = "X"
            paulis.append("".join(reversed(s)))
            coeffs.append(-h)

    return SparsePauliOp(paulis, np.array(coeffs, dtype=float))


def weighted_pauli_sum(terms: Iterable[Tuple[str, float]]) -> SparsePauliOp:
    """
    Build SparsePauliOp from explicit (pauli_string, coeff) pairs.
    Example: [("ZZII", -1.0), ("IXIX", -0.5)]
    """
    strings, coeffs = zip(*terms) if terms else ([], [])
    return SparsePauliOp(list(strings), np.asarray(coeffs, dtype=float))
