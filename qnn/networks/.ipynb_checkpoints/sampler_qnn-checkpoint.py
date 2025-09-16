# qnn/networks/sampler_qnn.py
from __future__ import annotations
from typing import Optional, Any
from qiskit_machine_learning.neural_networks import SamplerQNN

def make_sampler_qnn(
    circuit,
    sampler,
    interpret=None,
    output_shape=None,
    pass_manager: Optional[Any] = None,   
    input_gradients: bool = False,
):
    return SamplerQNN(
        circuit=circuit,
        interpret=interpret,
        output_shape=output_shape,
        sampler=sampler,
        pass_manager=pass_manager,         
        input_gradients=input_gradients,
    )