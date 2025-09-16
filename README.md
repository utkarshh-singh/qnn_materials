
# QNN Codebase (Parallel + Noise + Hardware)
- Parallel CV: `qnn/training/parallel.py` (joblib)
- Noisy Aer: `qnn/primitives/noise.py` + Aer primitives options
- Hardware Runtime: `qnn/primitives/runtime.py` helpers

## Quickstart
pip install -e .
python experiments/run_cls_estimator_aer_noise.py
python experiments/grid_search_parallel.py
# Hardware (after saving IBM Quantum token):
python experiments/run_cls_estimator_runtime.py
