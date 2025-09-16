# qnn/primitives/passes.py
from __future__ import annotations
from typing import Optional
from qiskit import transpile

def transpile_for_aer(circuit, optimization_level: int = 1, method: str = "automatic",
                      basis_gates: Optional[list[str]] = None, seed_transpiler: int = 42):
    from qiskit_aer import AerSimulator
    backend = AerSimulator(method=method)
    return transpile(circuit, backend=backend, basis_gates=basis_gates,
                     optimization_level=optimization_level, seed_transpiler=seed_transpiler)

def transpile_for_backend(circuit, backend, optimization_level: int = 1, seed_transpiler: int = 42):
    return transpile(circuit, backend=backend, optimization_level=optimization_level,
                     seed_transpiler=seed_transpiler)

def build_pass_manager(optimization_level: int = 1, target=None):
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    except Exception as exc:
        raise ImportError("Preset pass managers require a compatible qiskit version.") from exc
    return generate_preset_pass_manager(optimization_level=optimization_level, target=target)
