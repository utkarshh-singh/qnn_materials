# qnn/primitives/noise.py
from __future__ import annotations
from typing import Optional

# Aer noise
try:
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit_aer import AerSimulator
except Exception:
    NoiseModel = None
    depolarizing_error = None
    AerSimulator = None


# ---------- Fake backend resolver (no service required) ----------
def _get_fake_backend(name: str):
    """
    Resolve a fake backend by name, trying both modern and legacy locations.
    Examples (not exhaustive): 'fake_perth', 'fake_oslo', 'fake_jakarta', 'fake_vigo'
    """
    clean = name.strip().lower()
    # strip an optional "fake:" prefix for convenience
    if clean.startswith("fake:"):
        clean = clean[5:]

    candidates = []

    # Modern (Qiskit >= 0.45-ish): qiskit_ibm_runtime.fake_provider
    try:
        from qiskit_ibm_runtime.fake_provider import __all__ as _ALL_NEW  # type: ignore
        from qiskit_ibm_runtime import fake_provider as _NEW  # type: ignore
        candidates.append(("new", _NEW, {a.lower(): a for a in _ALL_NEW}))
    except Exception:
        pass

    # Legacy: qiskit.providers.fake_provider
    try:
        from qiskit.providers.fake_provider import __all__ as _ALL_OLD  # type: ignore
        import qiskit.providers.fake_provider as _OLD  # type: ignore
        candidates.append(("old", _OLD, {a.lower(): a for a in _ALL_OLD}))
    except Exception:
        pass

    # Common aliases map (so users can just say 'perth', 'oslo', etc.)
    aliases = {
        "perth": "FakePerth",
        "oslo": "FakeOslo",
        "jakarta": "FakeJakarta",
        "vigo": "FakeVigo",
        "quito": "FakeQuito",
        "manila": "FakeManila",
        "nairobi": "FakeNairobi",
        "lima": "FakeLima",
        "casablanca": "FakeCasablanca",
        "auckland": "FakeAuckland",
        "brisbane": "FakeBrisbane",
        "toronto": "FakeToronto",
        "melbourne": "FakeMelbourne",
        "rome": "FakeRome",
        "hanoi": "FakeHanoi",
    }

    # If the user gave just "oslo", upgrade to "FakeOslo"
    class_name_guess = aliases.get(clean, None)

    for tag, module, export_map in candidates:
        # direct match (e.g., "fakeoslo", "fake_oslo", "fakeoslov2") -> try exact key
        if clean in export_map:
            cls_name = export_map[clean]
            return getattr(module, cls_name)()
        # try canonical "FakeXxx" class name
        if class_name_guess and class_name_guess.lower() in export_map:
            cls_name = export_map[class_name_guess.lower()]
            return getattr(module, cls_name)()
        # try exact attribute lookups as last resort
        for try_name in (name, name.replace(" ", ""), name.replace("-", ""), class_name_guess or ""):
            if try_name:
                try:
                    return getattr(module, try_name)()
                except Exception:
                    pass

    raise ValueError(
        f"Could not resolve fake backend '{name}'. "
        "Try names like 'fake_oslo', 'FakeOslo', or a known device alias (e.g., 'oslo')."
    )


# ---------- Public helpers ----------
def basic_depolarizing_noise(p1: float = 1e-3, p2: float = 1e-2):
    if NoiseModel is None:
        raise ImportError("qiskit-aer not installed")
    nm = NoiseModel()
    e1 = depolarizing_error(p1, 1)
    e2 = depolarizing_error(p2, 2)
    # 1-qubit gates (add more if you like)
    for gate in ["id", "x", "y", "z", "h", "s", "sdg", "rx", "ry", "rz"]:
        nm.add_all_qubit_quantum_error(e1, gate)
    # 2-qubit gates
    for gate in ["cx", "cz"]:
        nm.add_all_qubit_quantum_error(e2, gate)
    return nm


def device_noise_model(
    *,
    backend=None,
    backend_name: Optional[str] = None,
    prefer_fake: bool = True,
):
    """
    Build an Aer NoiseModel from a device's calibrations—WITHOUT requiring a cloud service.

    Pass either:
      - `backend`: any backend-like object exposing calibration data
      - `backend_name`: the name of a FAKE backend (e.g., 'fake_oslo', 'oslo', 'FakeOslo')
        (set `prefer_fake=False` only if you are sure the name is a real backend you've already loaded)

    If both are None -> raises.
    """
    if NoiseModel is None:
        raise ImportError("qiskit-aer not installed")

    if backend is None:
        if backend_name is None:
            raise ValueError("Provide `backend` or `backend_name`.")
        # Resolve a fake backend locally (no account required)
        if prefer_fake:
            backend = _get_fake_backend(backend_name)
        else:
            raise ValueError(
                "prefer_fake=False requires a concrete `backend` object; "
                "this helper does not retrieve real backends."
            )

    return NoiseModel.from_backend(backend)


def aer_simulator_with_device_noise(
    *,
    backend=None,
    backend_name: Optional[str] = None,
    prefer_fake: bool = True,
):
    """
    Convenience: return an AerSimulator **configured from a (fake) device**,
    including noise model, coupling map, basis gates, etc.
    """
    if AerSimulator is None:
        raise ImportError("qiskit-aer not installed")

    if backend is None:
        if backend_name is None:
            raise ValueError("Provide `backend` or `backend_name`.")
        if prefer_fake:
            backend = _get_fake_backend(backend_name)
        else:
            raise ValueError(
                "prefer_fake=False requires a concrete `backend` object already in memory."
            )

    # This follows the Aer tutorial pattern:
    # AerSimulator.from_backend(...) pulls noise + configuration from the backend object.
    return AerSimulator.from_backend(backend)