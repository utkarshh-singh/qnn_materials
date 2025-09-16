# qnn/execution/executor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import hashlib, pickle
from joblib import Parallel, delayed

from qiskit.primitives import StatevectorEstimator, StatevectorSampler

# Prefer Aer V2, fallback to V1 if missing
try:
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2, SamplerV2 as AerSamplerV2
except Exception:
    AerEstimatorV2 = AerSamplerV2 = None
try:
    from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
except Exception:
    AerEstimator = AerSampler = None

from qiskit_aer import AerSimulator

from qnn.primitives.noise import basic_depolarizing_noise, device_noise_model

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except Exception:
    QiskitRuntimeService = None


ExecMode   = Literal["statevector", "aer", "runtime"]   # runtime kept minimal here
NoiseMode  = Literal["none", "depolarizing", "device", "custom"]  # custom = prebuilt noise model


@dataclass
class ExecutorConfig:
    mode: ExecMode = "statevector"
    shots: int = 4096
    seed: Optional[int] = 42
    aer_method: str = "automatic"
    optimization_level: int = 1
    n_jobs: int = 1
    cache_path: Optional[str] = None
    retries: int = 2

    # ---- noise config ----
    noise_mode: NoiseMode = "none"   # "none"|"depolarizing"|"device"|"custom"

    # depolarizing params
    p1: float = 1e-3
    p2: float = 1e-2

    # device noise (either provide backend object OR a name)
    device_backend: Optional[object] = None
    device_backend_name: Optional[str] = None
    service: Optional["QiskitRuntimeService"] = None
    # if device lookup fails, optionally fall back
    fallback_to_depolarizing: bool = True

    # custom prebuilt noise model (NoiseModel)
    noise_model: Optional[object] = None


class QNNExecutor:
    def __init__(self, cfg: ExecutorConfig):
        self.cfg = cfg
        self._cache: Dict[bytes, Any] = {}
        if cfg.cache_path:
            self._load_cache(cfg.cache_path)

    # ---------- noise resolver ----------
    def _resolve_noise_model(self):
        if self.cfg.noise_mode == "none":
            return None

        if self.cfg.noise_mode == "custom":
            return self.cfg.noise_model

        if self.cfg.noise_mode == "depolarizing":
            return basic_depolarizing_noise(self.cfg.p1, self.cfg.p2)

        if self.cfg.noise_mode == "device":
            # Prefer explicit backend object if provided
            try:
                if self.cfg.device_backend is not None:
                    return device_noise_model(backend=self.cfg.device_backend)
                if self.cfg.device_backend_name is not None:
                    svc = self.cfg.service
                    if svc is None and QiskitRuntimeService is not None:
                        svc = QiskitRuntimeService()
                    return device_noise_model(backend=None, backend_name=self.cfg.device_backend_name, service=svc)
                raise ValueError("device noise requested but neither `device_backend` nor `device_backend_name` provided.")
            except Exception as e:
                if self.cfg.fallback_to_depolarizing:
                    # graceful fallback, like your kernel style
                    return basic_depolarizing_noise(self.cfg.p1, self.cfg.p2)
                raise e

        raise ValueError(f"Unknown noise_mode: {self.cfg.noise_mode}")

    # ---------- primitives ----------
    def make_estimator(self):
        if self.cfg.mode == "statevector":
            return StatevectorEstimator()

        if self.cfg.mode == "aer":
            shots = self.cfg.shots
            seed  = self.cfg.seed
            noise_model = self._resolve_noise_model()

            if AerEstimatorV2 is not None:
                options = {
                    "backend_options": ({"noise_model": noise_model} if noise_model else {}),
                    "run_options": {"shots": shots, **({"seed": seed} if seed is not None else {})},
                }
                return AerEstimatorV2(options=options)

            assert AerEstimator is not None, "qiskit-aer not installed"
            return AerEstimator(
                backend_options={"noise_model": noise_model} if noise_model else None,
                run_options={"shots": shots, **({"seed": seed} if seed is not None else {})},
            )

        # runtime path: intentionally minimal
        raise NotImplementedError("Runtime path intentionally minimal for now.")

    def make_sampler(self):
        if self.cfg.mode == "statevector":
            return StatevectorSampler()

        if self.cfg.mode == "aer":
            shots = self.cfg.shots
            seed  = self.cfg.seed
            noise_model = self._resolve_noise_model()

            if AerSamplerV2 is not None:
                options = {
                    "backend_options": ({"noise_model": noise_model} if noise_model else {}),
                    "run_options": {"shots": shots, **({"seed": seed} if seed is not None else {})},
                }
                return AerSamplerV2(options=options)

            assert AerSampler is not None, "qiskit-aer not installed"
            return AerSampler(
                backend_options={"noise_model": noise_model} if noise_model else None,
                run_options={"shots": shots, **({"seed": seed} if seed is not None else {})},
            )

        raise NotImplementedError("Runtime path intentionally minimal for now.")

    # ---------- pass manager ----------
    def make_pass_manager(self, target=None):
        from qnn.primitives.passes import build_pass_manager
        return build_pass_manager(optimization_level=self.cfg.optimization_level, target=target)

    # ---------- parallel & cache ----------
    def map_parallel(self, items, fn):
        if self.cfg.n_jobs and self.cfg.n_jobs != 1:
            return Parallel(n_jobs=self.cfg.n_jobs)(delayed(fn)(x) for x in items)
        return [fn(x) for x in items]

    def _key(self, *objs) -> bytes:
        h = hashlib.sha256()
        for o in objs:
            h.update(pickle.dumps(o))
        return h.digest()

    def cached(self, key_objs, compute):
        key = self._key(*key_objs)
        if key in self._cache:
            return self._cache[key]
        last_err = None
        for _ in range(self.cfg.retries + 1):
            try:
                val = compute()
                self._cache[key] = val
                if self.cfg.cache_path:
                    self._save_cache(self.cfg.cache_path)
                return val
            except Exception as e:
                last_err = e
        raise last_err

    def _load_cache(self, path: str):
        try:
            with open(path, "rb") as f:
                self._cache = pickle.load(f)
        except Exception:
            self._cache = {}

    def _save_cache(self, path: str):
        try:
            with open(path, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception:
            pass
