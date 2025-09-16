# qnn/networks/estimator_qnn.py
from __future__ import annotations
from typing import Any, Optional
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp

def make_estimator_qnn(
    circuit,
    estimator,
    input_params=None,
    weight_params=None,
    observable: Any = "Z",
    pass_manager: Optional[Any] = None,     # <--- NEW: pass manager support
    default_precision: Optional[float] = None,
    input_gradients: bool = False,
):
    if isinstance(observable, str) and observable in {"Z", "X", "Y"}:
        obs = SparsePauliOp.from_list([(observable * circuit.num_qubits, 1.0)])
    else:
        obs = observable

    return EstimatorQNN(
        circuit=circuit,
        observables=obs,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
        pass_manager=pass_manager,            
        default_precision=default_precision,
        input_gradients=input_gradients,
    )
