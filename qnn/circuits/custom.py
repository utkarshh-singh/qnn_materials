# qnn/circuits/custom.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal, Tuple, List, Optional

from qiskit.circuit import QuantumCircuit, ParameterVector

EntType = Literal["cx_line", "cx_ring", "cz_line", "cz_ring", "full"]


@dataclass
class Entanglement:
    """Utility to apply a chosen entanglement pattern with a given 2q gate."""
    kind: EntType = "cx_line"

    def apply(self, qc: QuantumCircuit, layer_qubits: Optional[Iterable[int]] = None):
        qs = list(layer_qubits) if layer_qubits is not None else list(range(qc.num_qubits))
        if self.kind in {"cx_line", "cz_line"}:
            n = len(qs)
            r= [i for i in range(n + 1) if i % 2 == 0]
            for i in r[:-1]:
                ctrl, tgt = qs[i], qs[i + 1]
                if self.kind.startswith("cx"): qc.cx(ctrl, tgt)
                else: qc.cz(ctrl, tgt)
        elif self.kind in {"cx_ring", "cz_ring"}:
            # line + last-to-first
            self.__class__(self.kind.replace("ring", "line")).apply(qc, qs)
            ctrl, tgt = qs[-1], qs[0]
            if self.kind.startswith("cx"): qc.cx(ctrl, tgt)
            else: qc.cz(ctrl, tgt)
        elif self.kind == "full":
            # all-to-all CX in a simple triangular pattern
            for i in range(len(qs)):
                for j in range(i + 1, len(qs)):
                    qc.cx(qs[i], qs[j])
        else:
            raise ValueError(f"Unknown entanglement kind: {self.kind}")


# ----------------------- Encoders -----------------------

def yz_cx_encoding(num_qubits: int, reps: int = 1) -> Tuple[QuantumCircuit, ParameterVector]:
    """
    "YZ_CX_EncodingCircuit" style feature map:
      For each rep r:
        for each qubit i:  RY(x_i) -> RZ(x_i)
        entangle with CX in a line.
    Parameters are one per qubit (re-used each rep).  Uses only {RY,RZ,CX}.
    """
    x = ParameterVector("x", num_qubits)
    qc = QuantumCircuit(num_qubits, name="YZ_CX_Encode")
    ent = Entanglement("cx_line")

    for _ in range(reps):
        for i in range(num_qubits):
            qc.ry(x[i], i)
            qc.rz(x[i], i)
        ent.apply(qc)
    return qc, x


def raw_xz_encoding(num_qubits: int, reps: int = 1) -> Tuple[QuantumCircuit, ParameterVector]:
    """
    Simple XZ encoder:
      For each rep:
        RX(x_i) -> RZ(x_i), line CZ entanglement.
    """
    x = ParameterVector("x", num_qubits)
    qc = QuantumCircuit(num_qubits, name="XZ_Encode")
    ent = Entanglement("cz_line")
    for _ in range(reps):
        for i in range(num_qubits):
            qc.rx(x[i], i)
            qc.rz(x[i], i)
        ent.apply(qc)
    return qc, x


# ----------------------- Ansätze -----------------------

def hardware_efficient_ansatz(
    num_qubits: int,
    reps: int = 1,
    rot_sequence: Iterable[str] = ("ry", "rz"),   # per-qubit single-qubit rotations per rep
    entanglement: EntType = "cx_ring",
) -> Tuple[QuantumCircuit, ParameterVector]:
    """
    Minimal HEA using only {RX/RY/RZ, CX/CZ}.
    For each rep:
      - On each qubit apply rotations in 'rot_sequence' with fresh parameters.
      - Apply entanglement pattern.
    Returns (qc, theta) where len(theta) = reps * num_qubits * len(rot_sequence)
    """
    rot_seq = tuple(rot_sequence)
    theta = ParameterVector("theta", reps * num_qubits * len(rot_seq))
    qc = QuantumCircuit(num_qubits, name="HEA")
    ent = Entanglement(entanglement)

    k = 0
    for _ in range(reps):
        for q in range(num_qubits):
            for gate in rot_seq:
                if gate == "rx": qc.rx(theta[k], q)
                elif gate == "ry": qc.ry(theta[k], q)
                elif gate == "rz": qc.rz(theta[k], q)
                else: raise ValueError(f"Unsupported rotation '{gate}'. Use rx/ry/rz.")
                k += 1
        ent.apply(qc)
    return qc, theta


def layered_ansatz_with_barriers(
    num_qubits: int,
    reps: int = 2,
    rot_sequence: Iterable[str] = ("ry", "rz"),
    entanglement: EntType = "cx_line",
) -> Tuple[QuantumCircuit, ParameterVector]:
    """
    Same as HEA but with barriers between rotation and entanglement for readability/debug.
    """
    qc, theta = hardware_efficient_ansatz(num_qubits, reps, rot_sequence, entanglement)
    # rebuild with barriers (cheap way: re-generate)
    rot_seq = tuple(rot_sequence)
    qc2 = QuantumCircuit(num_qubits, name="HEA_Barriered")
    ent = Entanglement(entanglement)
    k = 0
    for _ in range(reps):
        for q in range(num_qubits):
            for gate in rot_seq:
                getattr(qc2, gate)(theta[k], q)  # rx/ry/rz
                k += 1
        qc2.barrier()
        ent.apply(qc2)
        qc2.barrier()
    return qc2, theta


# ----------------------- Composer -----------------------

def compose_encoder_ansatz(
    encoder: Tuple[QuantumCircuit, ParameterVector],
    ansatz: Tuple[QuantumCircuit, ParameterVector],
) -> Tuple[QuantumCircuit, List, List]:
    """
    Compose an encoder circuit U(x) with an ansatz V(θ):  V * U
    Returns (qc, input_params, weight_params)
    """
    enc, x = encoder
    ans, th = ansatz
    qc = QuantumCircuit(enc.num_qubits, name="QNN")
    qc.compose(enc, inplace=True)
    qc.compose(ans, inplace=True)
    return qc, list(x), list(th)
