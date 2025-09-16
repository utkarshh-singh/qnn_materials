import numpy as np
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.optimizers import COBYLA
from qnn.circuits.qnn_circuit import make_qnn_circuit
from qnn.networks.estimator_qnn import make_estimator_qnn
from qnn.models.classifier import make_classifier
from qnn.primitives.factory import make_estimator
from qnn.primitives.noise import basic_depolarizing_noise

algorithm_globals.random_seed = 42

num_inputs, num_samples = 2, 80
X = 2 * algorithm_globals.random.random([num_samples, num_inputs]) - 1
y01 = (np.sum(X, axis=1) >= 0).astype(int)
y = 2*y01 - 1

qc = make_qnn_circuit(num_qubits=num_inputs)

noise_model = basic_depolarizing_noise(1e-3, 1e-2)
estimator = make_estimator("aer", options={
    "shots": 4096,
    "seed": 42,
    "noise_model": noise_model,     # will be mapped to backend_options
})

qnn = make_estimator_qnn(qc, estimator=estimator)
clf = make_classifier(qnn, optimizer=COBYLA(maxiter=80))
clf.fit(X, y)
print("train accuracy (noisy Aer):", clf.score(X, y))