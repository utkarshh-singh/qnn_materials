# qnn/easy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Any, Tuple, Callable
from qiskit_machine_learning.optimizers import COBYLA
from qnn.training.callbacks import LiveObjectivePlot
import numpy as np

# Qiskit primitives
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
try:
    # Prefer Aer V2
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2, SamplerV2 as AerSamplerV2
except Exception:
    AerEstimatorV2 = AerSamplerV2 = None
try:
    # Fallback to Aer V1 (only if V2 missing)
    from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
except Exception:
    AerEstimator = AerSampler = None

# Our existing building blocks
from qnn.circuits.qnn_circuit import build_qnn_for_aer, build_qnn_raw
from qnn.networks.estimator_qnn import make_estimator_qnn
from qnn.networks.sampler_qnn import make_sampler_qnn
from qnn.models.classifier import make_classifier
try:
    from qnn.models.regressor import make_regressor
except Exception:
    make_regressor = None  # optional, depending on your repo structure

# Fake device helpers (service-free noise)
from qnn.primitives.fake_backends import get_fake_backend
from qiskit_aer.noise import NoiseModel

# Pass managers (auto-build if none is provided)
from qnn.primitives.passes import build_pass_manager

def _is_aer_primitive(prim: object) -> bool:
    mod = getattr(prim, "__module__", "") or getattr(type(prim), "__module__", "")
    return "qiskit_aer" in mod


Mode = Literal["simulation", "noisy_simulation"]
PrimitiveKind = Literal["auto", "estimator", "sampler"]


@dataclass
class EasyDefaults:
    shots: int = 2048
    seed: Optional[int] = 42
    optimization_level: int = 1
    decompose: bool = False            # you found decompose=True caused issues; keep False
    fake_backend_name: str = "oslo"    # used for noisy_simulation
    # feature map / ansatz knobs could be added here later


def _default_noise_model(fake_name: str) -> NoiseModel:
    """Build a device-like noise model from a local fake backend (no cloud)."""
    fb = get_fake_backend(fake_name)
    if fb is None:
        # try with 'Fake' prefix if short name was given
        fb = get_fake_backend("Fake" + fake_name.capitalize())
    if fb is None:
        raise RuntimeError(
            f"Could not find a fake backend for '{fake_name}'. "
            "Run discover_fake_backends() to see options."
        )
    return NoiseModel.from_backend(fb)


def _make_default_estimator(mode: Mode, shots: int, seed: Optional[int], defaults: EasyDefaults):
    """Return a ready-to-use Estimator based on mode (simulation / noisy_simulation)."""
    # Plain simulation: prefer Aer V2, fallback to Statevector if Aer missing
    if mode == "simulation":
        if AerEstimatorV2 is not None:
            return AerEstimatorV2(options={
                "run_options": {"shots": shots, **({"seed": seed} if seed is not None else {})},
            })
        # If Aer V2 unavailable, try Aer V1
        if AerEstimator is not None:
            return AerEstimator(run_options={"shots": shots, **({"seed": seed} if seed is not None else {})})
        # Final fallback: statevector (no shots)
        return StatevectorEstimator()

    # Noisy simulation (service-free via fake device)
    nm = _default_noise_model(defaults.fake_backend_name)
    if AerEstimatorV2 is not None:
        return AerEstimatorV2(options={
            "backend_options": {"noise_model": nm},
            "run_options": {"shots": shots, **({"seed": seed} if seed is not None else {})},
        })
    if AerEstimator is not None:
        return AerEstimator(
            backend_options={"noise_model": nm},
            run_options={"shots": shots, **({"seed": seed} if seed is not None else {})},
        )
    # If Aer missing entirely: you asked for noisy but can't do it → fall back to statevector
    return StatevectorEstimator()


def _make_default_sampler(mode: Mode, shots: int, seed: Optional[int], defaults: EasyDefaults):
    """Return a ready-to-use Sampler based on mode."""
    if mode == "simulation":
        if AerSamplerV2 is not None:
            return AerSamplerV2(options={"run_options": {"shots": shots, **({"seed": seed} if seed is not None else {})}})
        if AerSampler is not None:
            return AerSampler(run_options={"shots": shots, **({"seed": seed} if seed is not None else {})})
        return StatevectorSampler()

    nm = _default_noise_model(defaults.fake_backend_name)
    if AerSamplerV2 is not None:
        return AerSamplerV2(options={
            "backend_options": {"noise_model": nm},
            "run_options": {"shots": shots, **({"seed": seed} if seed is not None else {})},
        })
    if AerSampler is not None:
        return AerSampler(
            backend_options={"noise_model": nm},
            run_options={"shots": shots, **({"seed": seed} if seed is not None else {})},
        )
    return StatevectorSampler()


def _build_qnn(num_inputs: int, *, want_aer: bool, defaults: EasyDefaults):
    """
    Build a small QNN circuit; if want_aer=True we transpile/decompose for Aer
    to avoid 'unknown instruction' (e.g., ZZFeatureMap) errors.
    """
    if want_aer:
        # For Aer we *must* lower high-level library gates
        qc, in_p, wt_p = build_qnn_for_aer(
            num_qubits=num_inputs,
            optimization_level=defaults.optimization_level,
            decompose=True,   # <-- force decomposition for Aer
        )
    else:
        # Statevector is fine with high-level constructs
        qc, in_p, wt_p = build_qnn_raw(
            num_qubits=num_inputs,
            decompose=defaults.decompose,
        )
    return qc, in_p, wt_p


# ------------------------------- Public Facade -------------------------------

class QNNEasyClassifier:
    """
    Minimal, batteries-included classifier.

    Parameters
    ----------
    mode : {"simulation", "noisy_simulation"}
        Execution preset. "noisy_simulation" uses a fake-backend noise model (no cloud account needed).
    primitive : Optional[Estimator or Sampler]
        If provided, use this primitive directly. Otherwise we build a default one for `mode`.
    primitive_kind : {"auto","estimator","sampler"}
        Force how to wrap the QNN. "auto" picks estimator unless you pass a Sampler explicitly.
    shots, seed : ints
        Only used when we build Aer primitives.
    pass_manager : Optional[PassManager]
        If omitted, we build one with optimization_level=1.
    defaults : EasyDefaults
        Override package-wide defaults if you like (fake backend name, optimization level, etc.).
    """
    def __init__(
        self,
        mode: Mode = "simulation",
        *,
        primitive: Any = None,
        primitive_kind: PrimitiveKind = "auto",
        shots: int = EasyDefaults.shots,
        seed: Optional[int] = EasyDefaults.seed,
        pass_manager: Any = None,
        defaults: EasyDefaults = EasyDefaults(),
        callback = LiveObjectivePlot(),
        optimizer: Any = None, 
    ):
        self.mode = mode
        self.primitive = primitive
        self.primitive_kind = primitive_kind
        self.shots = shots
        self.seed = seed
        self.pass_manager = pass_manager or build_pass_manager(optimization_level=defaults.optimization_level)
        self.defaults = defaults
        self.optimizer = optimizer or COBYLA(maxiter=40)
        self.callback = callback
        self._clf = None  # sklearn-style classifier once built
        self._built_inputs = None  # num_inputs used to build circuit

    def _ensure_model(self, num_inputs: int):
        if self._clf is not None and self._built_inputs == num_inputs:
            return

        # decide sampler vs estimator
        use_sampler = False
        if self.primitive is not None:
            use_sampler = ("Sampler" in type(self.primitive).__name__)
        elif self.primitive_kind == "sampler":
            use_sampler = True

        # choose/build primitive
        prim = self.primitive
        if prim is None:
            prim = _make_default_sampler(self.mode, self.shots, self.seed, self.defaults) if use_sampler \
                else _make_default_estimator(self.mode, self.shots, self.seed, self.defaults)

        want_aer = _is_aer_primitive(prim)
        qc, in_params, wt_params = _build_qnn(num_inputs=num_inputs, want_aer=want_aer, defaults=self.defaults)

        # qnn + model
        if use_sampler:
            def interpret(bitstring: str):
                if not bitstring:
                    return 0
                return int(bitstring[-1])
            qnn = make_sampler_qnn(
                qc,
                sampler=prim,
                interpret=interpret,
                output_shape=2,
                pass_manager=self.pass_manager,
            )
        else:
            qnn = make_estimator_qnn(
                qc,
                estimator=prim,
                input_params=in_params,
                weight_params=wt_params,
                observable="Z",
                pass_manager=self.pass_manager,
            )

        self._clf = make_classifier(qnn, optimizer=self.optimizer, callback = self.callback)
        self._built_inputs = num_inputs


    # --- scikit-like API ---
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        self._ensure_model(num_inputs=X.shape[1])
        return self._clf.fit(X, y)

    def predict(self, X: np.ndarray):
        X = np.asarray(X)
        self._ensure_model(num_inputs=X.shape[1])
        return self._clf.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X); y = np.asarray(y)
        self._ensure_model(num_inputs=X.shape[1])
        return self._clf.score(X, y)


class QNNEasyRegressor:
    """
    Minimal, batteries-included regressor (Estimator-based by default).
    Same parameters as QNNEasyClassifier.
    """
    def __init__(
        self,
        mode: Mode = "simulation",
        *,
        primitive: Any = None,
        primitive_kind: PrimitiveKind = "auto",
        shots: int = EasyDefaults.shots,
        seed: Optional[int] = EasyDefaults.seed,
        pass_manager: Any = None,
        callback = LiveObjectivePlot(),
        defaults: EasyDefaults = EasyDefaults(),
        optimizer: Any = None
    ):
        if make_regressor is None:
            raise ImportError("make_regressor not available in your repo. Add it or switch to classifier.")
        self.mode = mode
        self.primitive = primitive
        self.primitive_kind = primitive_kind
        self.shots = shots
        self.seed = seed
        self.pass_manager = pass_manager or build_pass_manager(optimization_level=defaults.optimization_level)
        self.defaults = defaults
        self.optimizer = optimizer or COBYLA(maxiter=60)
        self._reg = None
        self.callback = callback
        self._built_inputs = None

    def _ensure_model(self, num_inputs: int):
        if self._reg is not None and self._built_inputs == num_inputs:
            return

        use_sampler = False
        if self.primitive is not None:
            use_sampler = ("Sampler" in type(self.primitive).__name__)
        elif self.primitive_kind == "sampler":
            use_sampler = True

        # choose/build primitive
        prim = self.primitive
        if prim is None:
            prim = _make_default_sampler(self.mode, self.shots, self.seed, self.defaults) if use_sampler \
                else _make_default_estimator(self.mode, self.shots, self.seed, self.defaults)

        # --- NEW: Aer-aware circuit build
        want_aer = _is_aer_primitive(prim)
        qc, in_params, wt_params = _build_qnn(num_inputs=num_inputs, want_aer=want_aer, defaults=self.defaults)

        # build QNN
        if use_sampler:
            # sampler-based regression is uncommon; keeping a simple interpret (last qubit)
            def interpret(bitstring: str):
                if not bitstring:
                    return 0
                return int(bitstring[-1])
            qnn = make_sampler_qnn(
                qc,
                sampler=prim,
                interpret=interpret,
                output_shape=2,
                pass_manager=self.pass_manager,
            )
        else:
            qnn = make_estimator_qnn(
                qc,
                estimator=prim,
                input_params=in_params,
                weight_params=wt_params,
                observable="Z",
                pass_manager=self.pass_manager,
            )

        self._reg = make_regressor(qnn, optimizer=self.optimizer, callback = self.callback)
        self._built_inputs = num_inputs

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X); y = np.asarray(y)
        self._ensure_model(num_inputs=X.shape[1])
        return self._reg.fit(X, y)

    def predict(self, X: np.ndarray):
        X = np.asarray(X)
        self._ensure_model(num_inputs=X.shape[1])
        return self._reg.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X); y = np.asarray(y)
        self._ensure_model(num_inputs=X.shape[1])
        return self._reg.score(X, y)
