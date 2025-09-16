
# QNN Codebase (Noise + Hardware)
- Noisy Aer: `qnn/primitives/noise.py` + Aer primitives options
- Hardware Runtime: `qnn/primitives/runtime.py` helpers

## Quickstart
- pip install -e .
- python experiments/run_cls_estimator_aer_noise.py
# Hardware (after saving IBM Quantum token):
- python experiments/run_cls_estimator_runtime.py
