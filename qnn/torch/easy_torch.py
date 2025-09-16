# qnn/torch/easy_torch.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal, Any, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Qiskit primitives
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
try:
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2, SamplerV2 as AerSamplerV2
except Exception:
    AerEstimatorV2 = AerSamplerV2 = None
try:
    from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
except Exception:
    AerEstimator = AerSampler = None

from qiskit_aer.noise import NoiseModel
from qiskit_machine_learning.connectors import TorchConnector

# Your internal helpers
from qnn.circuits.qnn_circuit import build_qnn_for_aer, build_qnn_raw
from qnn.primitives.fake_backends import get_fake_backend
from qnn.primitives.passes import build_pass_manager

Mode = Literal["simulation", "noisy_simulation"]
PrimitiveKind = Literal["auto", "estimator", "sampler"]

# ---------------------- Defaults ----------------------

@dataclass
class TorchEasyDefaults:
    shots: int = 2048
    seed: Optional[int] = 42
    optimization_level: int = 1
    fake_backend_name: str = "oslo"   # noise source for "noisy_simulation"
    decompose_for_statevector: bool = False  # keep high-level gates for sv

def _is_aer_primitive(prim: object) -> bool:
    mod = getattr(prim, "__module__", "") or getattr(type(prim), "__module__", "")
    return "qiskit_aer" in mod

def _noise_model_from_fake(name: str) -> NoiseModel:
    fb = get_fake_backend(name)
    if fb is None:
        # attempt canonical class
        fb = get_fake_backend("Fake" + name.capitalize())
    if fb is None:
        raise RuntimeError(
            f"Fake backend '{name}' not found. Run discover_fake_backends() to list options."
        )
    return NoiseModel.from_backend(fb)

def _make_default_estimator(mode: Mode, defaults: TorchEasyDefaults):
    if mode == "simulation":
        if AerEstimatorV2 is not None:
            return AerEstimatorV2(options={
                "run_options": {"shots": defaults.shots, **({"seed": defaults.seed} if defaults.seed is not None else {})},
            })
        if AerEstimator is not None:
            return AerEstimator(
                run_options={"shots": defaults.shots, **({"seed": defaults.seed} if defaults.seed is not None else {})}
            )
        return StatevectorEstimator()

    # noisy_simulation
    nm = _noise_model_from_fake(defaults.fake_backend_name)
    if AerEstimatorV2 is not None:
        return AerEstimatorV2(options={
            "backend_options": {"noise_model": nm},
            "run_options": {"shots": defaults.shots, **({"seed": defaults.seed} if defaults.seed is not None else {})},
        })
    if AerEstimator is not None:
        return AerEstimator(
            backend_options={"noise_model": nm},
            run_options={"shots": defaults.shots, **({"seed": defaults.seed} if defaults.seed is not None else {})},
        )
    # fall back
    return StatevectorEstimator()

def _make_default_sampler(mode: Mode, defaults: TorchEasyDefaults):
    if mode == "simulation":
        if AerSamplerV2 is not None:
            return AerSamplerV2(options={
                "run_options": {"shots": defaults.shots, **({"seed": defaults.seed} if defaults.seed is not None else {})},
            })
        if AerSampler is not None:
            return AerSampler(
                run_options={"shots": defaults.shots, **({"seed": defaults.seed} if defaults.seed is not None else {})}
            )
        return StatevectorSampler()

    nm = _noise_model_from_fake(defaults.fake_backend_name)
    if AerSamplerV2 is not None:
        return AerSamplerV2(options={
            "backend_options": {"noise_model": nm},
            "run_options": {"shots": defaults.shots, **({"seed": defaults.seed} if defaults.seed is not None else {})},
        })
    if AerSampler is not None:
        return AerSampler(
            backend_options={"noise_model": nm},
            run_options={"shots": defaults.shots, **({"seed": defaults.seed} if defaults.seed is not None else {})},
        )
    return StatevectorSampler()

def _build_qnn(num_inputs: int, *, want_aer: bool, defaults: TorchEasyDefaults):
    """
    For Aer primitives we MUST decompose (to avoid ZZFeatureMap errors).
    For statevector, we can keep high-level gates.
    """
    if want_aer:
        return build_qnn_for_aer(
            num_qubits=num_inputs,
            optimization_level=defaults.optimization_level,
            decompose=True,
        )
    return build_qnn_raw(
        num_qubits=num_inputs,
        decompose=defaults.decompose_for_statevector,
    )

# ---------------------- Torch Modules ----------------------

@dataclass
class TorchQNNConfig:
    num_inputs: int
    mode: Mode = "simulation"
    primitive: Any = None
    primitive_kind: PrimitiveKind = "auto"
    observable: str = "Z"
    interpret: Optional[Callable[[str], Optional[int]]] = None
    output_shape: Optional[int] = None
    pass_manager: Any = None
    defaults: TorchEasyDefaults = field(default_factory=TorchEasyDefaults)  # <-- fix
    dtype: torch.dtype = torch.float32
    initial_weights: Optional[torch.Tensor] = None

class QNNTorchEstimator(nn.Module):
    """Torch nn.Module wrapping an EstimatorQNN via TorchConnector (binary/regression)."""
    def __init__(self, cfg: TorchQNNConfig):
        super().__init__()
        self.cfg = cfg

        # choose primitive
        prim = cfg.primitive or _make_default_estimator(cfg.mode, cfg.defaults)
        want_aer = _is_aer_primitive(prim)

        # circuit
        qc, in_params, wt_params = _build_qnn(cfg.num_inputs, want_aer=want_aer, defaults=cfg.defaults)
        pm = cfg.pass_manager or build_pass_manager(optimization_level=cfg.defaults.optimization_level)

        # build EstimatorQNN
        from qnn.networks.estimator_qnn import make_estimator_qnn
        qnn = make_estimator_qnn(
            qc,
            estimator=prim,
            input_params=in_params,
            weight_params=wt_params,
            observable=cfg.observable,
            pass_manager=pm,
            input_gradients=True,
        )

        init_w = cfg.initial_weights.detach().clone().to(dtype=cfg.dtype) if cfg.initial_weights is not None else None
        self.qnn_torch = TorchConnector(qnn, initial_weights=init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # enforce input dtype
        x = x.to(dtype=self.cfg.dtype)
        out = self.qnn_torch(x)
        # ensure output has consistent shape/dtype
        if out.ndim == 1:
            out = out.unsqueeze(-1)
        return out.to(dtype=self.cfg.dtype)


class QNNTorchSampler(nn.Module):
    """Torch nn.Module wrapping a SamplerQNN via TorchConnector (e.g., multiclass)."""
    def __init__(self, cfg: TorchQNNConfig):
        super().__init__()
        self.cfg = cfg

        prim = cfg.primitive or _make_default_sampler(cfg.mode, cfg.defaults)
        want_aer = _is_aer_primitive(prim)

        qc, in_params, wt_params = _build_qnn(cfg.num_inputs, want_aer=want_aer, defaults=cfg.defaults)
        pm = cfg.pass_manager or build_pass_manager(optimization_level=cfg.defaults.optimization_level)

        # default binary interpret (last qubit) if none provided
        interpret = cfg.interpret
        out_shape = cfg.output_shape
        if interpret is None and out_shape is None:
            def interpret(bitstring: str) -> Optional[int]:
                return int(bitstring[-1]) if bitstring else 0
            out_shape = 2

        from qnn.networks.sampler_qnn import make_sampler_qnn
        qnn = make_sampler_qnn(
            qc,
            sampler=prim,
            interpret=interpret,
            output_shape=out_shape,
            pass_manager=pm,
            input_gradients=False,
        )

        init_w = cfg.initial_weights.detach().clone().to(dtype=cfg.dtype) if cfg.initial_weights is not None else None
        self.qnn_torch = TorchConnector(qnn, initial_weights=init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.cfg.dtype)
        out = self.qnn_torch(x)
        return out.to(dtype=self.cfg.dtype)

