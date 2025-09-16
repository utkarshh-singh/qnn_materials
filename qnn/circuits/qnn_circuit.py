# qnn/circuits/qnn_circuit.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, List
from qiskit.circuit import QuantumCircuit
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def _make_qnnc(
    num_qubits: int,
    feature_map,
    ansatz,
    fm_kwargs: Optional[Dict[str, Any]],
    ansatz_kwargs: Optional[Dict[str, Any]],
) -> QNNCircuit:
    fm = (feature_map or ZZFeatureMap)(num_qubits, **(fm_kwargs or {}))
    an = (ansatz or RealAmplitudes)(num_qubits, **(ansatz_kwargs or {}))
    return QNNCircuit(num_qubits=num_qubits, feature_map=fm, ansatz=an)

def _capture_params(qnnc: QNNCircuit):
    return list(qnnc.input_parameters), list(qnnc.weight_parameters)

def build_qnn_raw(
    num_qubits: int,
    feature_map=None,
    ansatz=None,
    fm_kwargs: Optional[Dict[str, Any]] = None,
    ansatz_kwargs: Optional[Dict[str, Any]] = None,
    *,
    decompose: bool = False,                        # <--- default False
) -> Tuple[QuantumCircuit, List, List]:
    qnnc = _make_qnnc(num_qubits, feature_map, ansatz, fm_kwargs, ansatz_kwargs)
    in_params, wt_params = _capture_params(qnnc)
    qc: QuantumCircuit = qnnc
    if decompose:
        qc = qc.decompose(reps=3)
    return qc, in_params, wt_params

def build_qnn_for_aer(
    num_qubits: int,
    feature_map=None,
    ansatz=None,
    fm_kwargs: Optional[Dict[str, Any]] = None,
    ansatz_kwargs: Optional[Dict[str, Any]] = None,
    *,
    optimization_level: int = 1,
    method: str = "automatic",
    basis_gates: Optional[list[str]] = None,
    seed_transpiler: int = 42,
    decompose: bool = True
) -> Tuple[QuantumCircuit, List, List]:
    # from qnn.primitives.passes import transpile_for_aer
    qc, in_params, wt_params = build_qnn_raw(
        num_qubits,
        feature_map=feature_map,
        ansatz=ansatz,
        fm_kwargs=fm_kwargs,
        ansatz_kwargs=ansatz_kwargs,
        decompose=decompose,
    )
    qc = qc.decompose(reps=3)
    # qc = transpile_for_aer(
    #     qc,
    #     optimization_level=optimization_level,
    #     method=method,
    #     basis_gates=basis_gates,
    #     seed_transpiler=seed_transpiler,
    # )
    return qc, in_params, wt_params

def build_qnn_for_runtime(
    num_qubits: int,
    feature_map=None,
    ansatz=None,
    fm_kwargs: Optional[Dict[str, Any]] = None,
    ansatz_kwargs: Optional[Dict[str, Any]] = None,
    *,
    backend=None,
    optimization_level: int = 1,
    seed_transpiler: int = 42,
    pass_manager=None,
    decompose: bool = False,
    target=None,
) -> Tuple[QuantumCircuit, List, List]:
    qc, in_params, wt_params = build_qnn_raw(
        num_qubits,
        feature_map=feature_map,
        ansatz=ansatz,
        fm_kwargs=fm_kwargs,
        decompose=decompose,
        ansatz_kwargs=ansatz_kwargs
    )

    if backend is not None:
        pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)
        qc = pm.run(qc)
    else:
        pm = pass_manager or generate_preset_pass_manager(optimization_level=optimization_level, target=target)
        qc = pm.run(qc)

    return qc, in_params, wt_params