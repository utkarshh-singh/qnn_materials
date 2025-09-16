# Quantum Neural Network (QNN) Package

# QNN Codebase (Noise + Hardware)
- Noisy Aer: `qnn/primitives/noise.py` + Aer primitives options
- Hardware Runtime: `qnn/primitives/runtime.py` helpers

## Quickstart
- pip install -e .
- python experiments/run_cls_estimator_aer_noise.py
# Hardware (after saving IBM Quantum token):
- python experiments/run_cls_estimator_runtime.py
=======
This package provides a **modular and user-friendly interface** to design and train **Quantum Neural Networks (QNNs)** on simulators, noisy backends, and hardware.  

It wraps the power of **Qiskit Machine Learning**, **Aer**, and **IBM Runtime** into a flexible but beginner-friendly API.

---

## ⚡ Quickstart in 3 lines

```python
from qnn.easy import QNNEasyClassifier

clf = QNNEasyClassifier(mode="simulation").fit(X, y)
print(clf.score(X, y))
```

---

## ✨ Features

- **Easy mode**: Just call `QNNEasyClassifier` or `QNNEasyRegressor` with `mode="simulation"` or `mode="noisy_simulation"`. Default options are provided for all parameters.
- **Custom circuits**: Prebuilt encoders and ansätze (`YZ_CX_EncodingCircuit`, `hardware efficient ansatz`, etc.) that are Aer-friendly.
- **Custom observables**: Ising Hamiltonians, parity operators, or single-qubit Paulis as observables for `EstimatorQNN`.
- **Noise support**: Plug-and-play noisy simulations with:
  - Aer device noise models (`NoiseModel.from_backend`).
  - Fake backends (`FakeOslo`, `FakeAthensV2`, etc.).
- **Torch connector**: Wrap your QNN into a PyTorch module (`TorchConnector`) for seamless integration with deep learning workflows.
- **Parallel execution**: Joblib-powered parallelization for kernel/QNN matrix building.
- **Safe execution**: Design inspired by `QuantumExecutor` in kernels — ensures that jobs can be resumed and noise models are handled safely.

---

## 📂 Repository Structure

```
qnn_codebase/
│
├── qnn/
│   ├── circuits/
│   │   ├── custom.py        # Encoders + ansätze builders (YZ_CX, HEA, etc.)
│   ├── observables/
│   │   ├── custom.py        # Ising Hamiltonians, parity-Z, Pauli ops
│   ├── networks/
│   │   ├── estimator_qnn.py # Make EstimatorQNN with pass_manager + observables
│   │   ├── sampler_qnn.py   # Make SamplerQNN
│   ├── execution/
│   │   ├── executor.py      # Unified execution logic for Aer/Runtime
│   │   ├── noise.py         # Noise model utilities (device/fake backends)
│   ├── easy.py              # QNNEasyClassifier / QNNEasyRegressor (defaults)
│   ├── torch/
│   │   ├── easy_torch.py    # TorchConnector integration
│   └── __init__.py
│
├── examples/
│   ├── classifier_demo.ipynb
│   ├── regressor_demo.ipynb
│   └── torch_demo.ipynb
│
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install qiskit qiskit-machine-learning qiskit-aer qiskit-ibm-runtime torch joblib tqdm
```

### 2. Basic usage: Easy Classifier

```python
import numpy as np
from qnn.easy import QNNEasyClassifier

# toy dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])   # XOR

# plain simulation (no noise)
clf = QNNEasyClassifier(mode="simulation")
clf.fit(X, y)
print("acc (sim):", clf.score(X, y))

# noisy simulation with fake backend
clf_noisy = QNNEasyClassifier(mode="noisy_simulation", device_backend="FakeOslo")
clf_noisy.fit(X, y)
print("acc (noisy):", clf_noisy.score(X, y))
```

---

### 3. Build a custom QNN (EstimatorQNN)

```python
from qnn.circuits.custom import yz_cx_encoding, hardware_efficient_ansatz, compose_encoder_ansatz
from qnn.observables.custom import ising_hamiltonian
from qnn.networks.estimator_qnn import make_estimator_qnn
from qnn.execution.executor import QNNExecutor

num_qubits = 4
enc = yz_cx_encoding(num_qubits, reps=1)
ans = hardware_efficient_ansatz(num_qubits, reps=2, entanglement="cx_ring")
qc, in_params, wt_params = compose_encoder_ansatz(enc, ans)

# observable: Ising Hamiltonian
obs = ising_hamiltonian(num_qubits, J=1.0, h=0.2, topology="ring")

# executor (Aer with noise support)
exe = QNNExecutor(mode="noisy_simulation", device_backend="FakeOslo")

estimator = exe.make_estimator()
qnn = make_estimator_qnn(
    qc,
    estimator=estimator,
    input_params=in_params,
    weight_params=wt_params,
    observable=obs,
)

print("QNN built with:", qnn)
```

---

### 4. PyTorch integration

```python
import torch
import torch.nn as nn
import torch.optim as optim
from qnn.torch.easy_torch import QNNTorchEstimator, TorchQNNConfig

# XOR dataset
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([0,1,1,0], dtype=torch.float32).unsqueeze(-1)

# Torch QNN model
model = QNNTorchEstimator(TorchQNNConfig(num_inputs=2, mode="simulation"))
criterion = nn.BCEWithLogitsLoss()
opt = optim.Adam(model.parameters(), lr=0.1)

# training loop
for epoch in range(20):
    opt.zero_grad()
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    opt.step()
    print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")
```

---

## 🧩 Available Components

### Encoders
- `yz_cx_encoding(num_qubits, reps)` – RY+RZ with CX entanglement
- `raw_xz_encoding(num_qubits, reps)` – RX+RZ with CZ entanglement

### Ansätze
- `hardware_efficient_ansatz(num_qubits, reps, rot_sequence, entanglement)`
- `layered_ansatz_with_barriers(...)`

### Observables
- `z_on(i, n)` – Z on qubit *i*
- `parity_z(n)` – global parity operator
- `ising_hamiltonian(n, J, h, topology)` – transverse-field Ising
- `weighted_pauli_sum(terms)` – arbitrary custom sums

---

## 🛠 Roadmap

- [ ] Add `factory.py` for one-line QNN builders (`build_qnn(num_qubits, encoder="yzcx", ansatz="hea", obs="ising")`).
- [ ] Add resumption and job management for hardware runs (like kernel executor).
- [ ] Extend to SamplerQNN (probabilistic models).
- [ ] Add visualization tools for circuits and observables.

---

## 📖 References

- [Qiskit Machine Learning Docs](https://qiskit-community.github.io/qiskit-machine-learning/)
- [Qiskit Aer Noise Simulation Tutorial](https://qiskit.github.io/qiskit-aer/tutorials/2_device_noise_simulation.html)
- Schuld & Killoran (2019). *Quantum Machine Learning in Feature Hilbert Spaces.*
- Havlíček et al. (2019). *Supervised learning with quantum-enhanced feature spaces.*

---

## 🙌 Acknowledgments

This package design was inspired by research and development in **quantum kernels** and **quantum reservoir computing**, extended here for practical QNN workflows.
