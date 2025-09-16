# qnn/torch/connector.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any, Callable, Literal

import torch
import torch.nn as nn
from qiskit_machine_learning.connectors import TorchConnector

from qnn.execution.executor import QNNExecutor, ExecutorConfig
from qnn.circuits.qnn_circuit import build_qnn_for_aer, build_qnn_raw
from qnn.primitives.passes import build_pass_manager

# Qiskit primitives (type hints only)
try:
    from qiskit.primitives.base import BaseEstimator, BaseSampler  # type: ignore
except Exception:
    BaseEstimator = BaseSampler = Any

Mode = Literal["statevector", "aer", "runtime"]

def _is_aer_mode(exec_cfg: ExecutorConfig | None) -> bool:
    return exec_cfg is not None and exec_cfg.mode == "aer"

@dataclass
class EstimatorTorchConfig:
    """
    Full-control config for a Torch module backed by EstimatorQNN.
    """
    num_inputs: int
    # execution
    exec_cfg: Optional[ExecutorConfig] = None              # if None -> statevector
    estimator: Optional[BaseEstimator] = None              # if given, overrides exec_cfg
    # circuit build
    decompose_for_aer: bool = True                         # avoid ZZFeatureMap in Aer
    optimization_level: int = 1
    pass_manager: Any = None
    # qnn specifics
    observable: Any = "Z"
    input_gradients: bool = True
    # torch
    initial_weights: Optional[torch.Tensor] = None
    dtype: torch.dtype = torch.float32

class QNNEstimatorTorch(nn.Module):
    """
    Torch nn.Module that wraps an EstimatorQNN via TorchConnector.
    Outputs expectation values in [-1, 1] (shape: [batch, 1]).
    """
    def __init__(self, cfg: EstimatorTorchConfig):
        super().__init__()
        self.cfg = cfg

        # 1) primitive
        if cfg.estimator is not None:
            estimator = cfg.estimator
            want_aer = True  # assume user passed an Aer estimator
        else:
            exec_cfg = cfg.exec_cfg or ExecutorConfig(mode="statevector")
            estimator = QNNExecutor(exec_cfg).make_estimator()
            want_aer = _is_aer_mode(exec_cfg)

        # 2) circuit
        if want_aer and cfg.decompose_for_aer:
            qc, in_params, wt_params = build_qnn_for_aer(
                num_qubits=cfg.num_inputs,
                optimization_level=cfg.optimization_level,
                decompose=True,
            )
        else:
            qc, in_params, wt_params = build_qnn_raw(
                num_qubits=cfg.num_inputs,
                decompose=False,
            )

        # 3) pass manager
        pm = cfg.pass_manager or build_pass_manager(optimization_level=cfg.optimization_level)

        # 4) EstimatorQNN
        from qnn.networks.estimator_qnn import make_estimator_qnn
        qnn = make_estimator_qnn(
            qc,
            estimator=estimator,
            input_params=in_params,
            weight_params=wt_params,
            observable=cfg.observable,
            pass_manager=pm,
            input_gradients=cfg.input_gradients,
        )

        init_w = None
        if cfg.initial_weights is not None:
            init_w = cfg.initial_weights.detach().clone().to(dtype=cfg.dtype)

        self.qnn_torch = TorchConnector(qnn, initial_weights=init_w, dtype=cfg.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.qnn_torch(x)
        if out.ndim == 1:
            out = out.unsqueeze(-1)
        return out
        

@dataclass
class SamplerTorchConfig:
    """
    Full-control config for a Torch module backed by SamplerQNN.
    """
    num_inputs: int
    # execution
    exec_cfg: Optional[ExecutorConfig] = None          # if None -> statevector
    sampler: Optional[BaseSampler] = None              # if given, overrides exec_cfg
    # circuit build
    decompose_for_aer: bool = True
    optimization_level: int = 1
    pass_manager: Any = None
    # qnn specifics
    interpret: Optional[Callable[[str], Optional[int]]] = None
    output_shape: Optional[int] = None
    input_gradients: bool = False
    # torch
    initial_weights: Optional[torch.Tensor] = None
    dtype: torch.dtype = torch.float32

class QNNSamplerTorch(nn.Module):
    """
    Torch nn.Module that wraps a SamplerQNN via TorchConnector.
    Output shape depends on interpret/output_shape.
    """
    def __init__(self, cfg: SamplerTorchConfig):
        super().__init__()
        self.cfg = cfg

        # 1) primitive
        if cfg.sampler is not None:
            sampler = cfg.sampler
            want_aer = True
        else:
            exec_cfg = cfg.exec_cfg or ExecutorConfig(mode="statevector")
            sampler = QNNExecutor(exec_cfg).make_sampler()
            want_aer = _is_aer_mode(exec_cfg)

        # 2) circuit
        if want_aer and cfg.decompose_for_aer:
            qc, in_params, wt_params = build_qnn_for_aer(
                num_qubits=cfg.num_inputs,
                optimization_level=cfg.optimization_level,
                decompose=True,
            )
        else:
            qc, in_params, wt_params = build_qnn_raw(
                num_qubits=cfg.num_inputs,
                decompose=False,
            )

        # 3) pass manager
        pm = cfg.pass_manager or build_pass_manager(optimization_level=cfg.optimization_level)

        # 4) SamplerQNN
        from qnn.networks.sampler_qnn import make_sampler_qnn
        interpret = cfg.interpret
        out_shape = cfg.output_shape
        if interpret is None and out_shape is None:
            # default: last-qubit binary mapping
            def interpret(bitstring: str) -> Optional[int]:
                return int(bitstring[-1]) if bitstring else 0
            out_shape = 2

        qnn = make_sampler_qnn(
            qc,
            sampler=sampler,
            interpret=interpret,
            output_shape=out_shape,
            pass_manager=pm,
            input_gradients=cfg.input_gradients,
        )

        init_w = None
        if cfg.initial_weights is not None:
            init_w = cfg.initial_weights.detach().clone().to(dtype=cfg.dtype)

        self.qnn_torch = TorchConnector(qnn, initial_weights=init_w, dtype=cfg.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnn_torch(x)
