# qnn/primitives/factory.py
from __future__ import annotations
from typing import Literal, Dict, Any

from qiskit.primitives import StatevectorEstimator, StatevectorSampler

# Aer imports are optional
try:
    from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
except Exception:
    AerEstimator = AerSampler = None

# Optional V2 imports (newer API)
try:
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2, SamplerV2 as AerSamplerV2
except Exception:
    AerEstimatorV2 = AerSamplerV2 = None


def _split_aer_options(options: Dict[str, Any] | None):
    """Split combined options dict into backend/run/transpile buckets."""
    options = options or {}
    backend_options = dict(options.get("backend_options", {}))
    run_options = dict(options.get("run_options", {}))

    if "seed" in options:
        run_options.setdefault("seed", options["seed"])
    if "shots" in options:
        run_options.setdefault("shots", options["shots"])
    if "noise_model" in options:
        backend_options.setdefault("noise_model", options["noise_model"])

    return backend_options, run_options


def make_estimator(kind: Literal["statevector", "aer", "runtime"] = "statevector",
                   **kwargs):
    if kind == "statevector":
        return StatevectorEstimator()

    if kind == "aer":
        assert (AerEstimator is not None) or (AerEstimatorV2 is not None), "qiskit-aer not installed"

        options = kwargs.get("options", {})
        backend_options, run_options = _split_aer_options(options)

        # Prefer V2 if available (unified options)
        if AerEstimatorV2 is not None:
            return AerEstimatorV2(options={
                "backend_options": backend_options,
                "run_options": run_options,
            })
        # Fallback to V1 (split kwargs)
        return AerEstimator(
            backend_options=backend_options or None,
            run_options=run_options or None,
        )

    if kind == "runtime":
        if "runtime_estimator" not in kwargs:
            raise ValueError("Provide runtime_estimator from qnn.primitives.runtime.make_runtime_estimator")
        return kwargs["runtime_estimator"]

    raise ValueError(kind)


def make_sampler(kind: Literal["statevector", "aer", "runtime"] = "statevector",
                 **kwargs):
    if kind == "statevector":
        return StatevectorSampler()

    if kind == "aer":
        assert (AerSampler is not None) or (AerSamplerV2 is not None), "qiskit-aer not installed"

        options = kwargs.get("options", {})
        backend_options, run_options = _split_aer_options(options)

        if AerSamplerV2 is not None:
            return AerSamplerV2(options={
                "backend_options": backend_options,
                "run_options": run_options,
            })
        return AerSampler(
            backend_options=backend_options or None,
            run_options=run_options or None,
        )

    if kind == "runtime":
        if "runtime_sampler" not in kwargs:
            raise ValueError("Provide runtime_sampler from qnn.primitives.runtime.make_runtime_sampler")
        return kwargs["runtime_sampler"]

    raise ValueError(kind)
