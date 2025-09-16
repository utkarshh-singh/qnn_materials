# qnn/primitives/fake_device.py
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
from qnn.primitives.fake_backends import get_fake_backend

def noise_model_from_fake(name: str) -> NoiseModel:
    """
    Build an Aer NoiseModel using a local fake backend (no service).
    """
    fb = get_fake_backend(name)
    if fb is None:
        raise ValueError(f"Fake backend '{name}' not found. Run discover_fake_backends() to list options.")
    return NoiseModel.from_backend(fb)

def aer_simulator_from_fake(name: str) -> AerSimulator:
    """
    Build an AerSimulator configured from a fake backend:
    includes noise, coupling map, basis gates, etc.
    """
    fb = get_fake_backend(name)
    if fb is None:
        raise ValueError(f"Fake backend '{name}' not found.")
    return AerSimulator.from_backend(fb)
