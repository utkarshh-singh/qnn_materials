# qnn/primitives/runtime.py
from __future__ import annotations
from typing import Optional, Tuple, Literal
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    Session,
    EstimatorV2 as RuntimeEstimator,
    SamplerV2 as RuntimeSampler,
)
from qiskit_ibm_runtime.options import EstimatorOptions, SamplerOptions

# def make_runtime_estimator(backend_name, service=None, shots: int=4096, optimization_level: int=1, resilience_level: int=0):
#     service = service or QiskitRuntimeService()
#     backend = service.backend(backend_name)
#     session = Session(backend=backend)
#     est = RuntimeEstimator(session=session, options={
#         "resilience_level": resilience_level,
#         "optimization_level": optimization_level,
#         "default_shots": shots,
#     })
#     return est, session

# def make_runtime_sampler(backend_name, service=None, shots: int=4096, optimization_level: int=1, resilience_level: int=0):
#     service = service or QiskitRuntimeService()
#     backend = service.backend(backend_name)
#     session = Session(backend=backend)
#     smp = RuntimeSampler(session=session, options={
#         "resilience_level": resilience_level,
#         "optimization_level": optimization_level,
#         "default_shots": shots,
#     })
#     return smp, session

ExecutionMode = Literal["auto", "job", "session"]


def _estimator_options(
    shots: int,
    resilience_level: Optional[int] = None,
    seed: Optional[int] = None,
) -> EstimatorOptions:
    """
    Build EstimatorOptions for V2 primitives.
    Valid keys include: default_shots, resilience_level, seed_estimator, execution, ...
    Docs: https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/options-estimator-options
    """
    opts = EstimatorOptions()
    opts.default_shots = shots
    if resilience_level is not None:
        opts.resilience_level = resilience_level
    if seed is not None:
        opts.seed_estimator = seed
    return opts


def _sampler_options(
    shots: int,
) -> SamplerOptions:
    """
    Build SamplerOptions for V2 primitives.
    Valid keys include: default_shots, execution, ...
    Docs: https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/options-sampler-options
    """
    opts = SamplerOptions()
    opts.default_shots = shots
    return opts


def make_runtime_estimator(
    backend_name: str,
    service: Optional[QiskitRuntimeService] = None,
    shots: int = 1024,
    resilience_level: Optional[int] = None,
    seed: Optional[int] = None,
    execution_mode: ExecutionMode = "auto",
) -> Tuple[RuntimeEstimator, Optional[Session]]:
    """
    Returns (estimator, session). In job mode, session is None.

    execution_mode:
      - 'job'     : Open Plan compatible (no sessions). Uses mode=backend.
      - 'session' : Requires a plan with session support. Uses mode=Session(backend).
      - 'auto'    : Try session; on authorization error (code 1352) fall back to job.
    """
    service = service or QiskitRuntimeService()
    backend = service.backend(backend_name)
    options = _estimator_options(shots, resilience_level, seed)

    if execution_mode == "job":
        return RuntimeEstimator(mode=backend, options=options), None

    if execution_mode == "session":
        sess = Session(backend=backend)
        return RuntimeEstimator(mode=sess, options=options), sess

    # auto
    try:
        sess = Session(backend=backend)
        return RuntimeEstimator(mode=sess, options=options), sess
    except Exception as exc:
        msg = str(exc)
        if "code\":1352" in msg or "not authorized to run a session" in msg.lower():
            # Open Plan fallback
            return RuntimeEstimator(mode=backend, options=options), None
        raise


def make_runtime_sampler(
    backend_name: str,
    service: Optional[QiskitRuntimeService] = None,
    shots: int = 1024,
    execution_mode: ExecutionMode = "auto",
) -> Tuple[RuntimeSampler, Optional[Session]]:
    """
    Returns (sampler, session). In job mode, session is None.
    """
    service = service or QiskitRuntimeService()
    backend = service.backend(backend_name)
    options = _sampler_options(shots)

    if execution_mode == "job":
        return RuntimeSampler(mode=backend, options=options), None

    if execution_mode == "session":
        sess = Session(backend=backend)
        return RuntimeSampler(mode=sess, options=options), sess

    # auto
    try:
        sess = Session(backend=backend)
        return RuntimeSampler(mode=sess, options=options), sess
    except Exception as exc:
        msg = str(exc)
        if "code\":1352" in msg or "not authorized to run a session" in msg.lower():
            return RuntimeSampler(mode=backend, options=options), None
        raise
