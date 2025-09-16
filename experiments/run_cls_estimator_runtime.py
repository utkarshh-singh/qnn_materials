
import numpy as np
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.optimizers import COBYLA
from qnn.circuits.qnn_circuit import make_qnn_circuit
from qnn.networks.estimator_qnn import make_estimator_qnn
from qnn.models.classifier import make_classifier
from qnn.primitives.runtime import make_runtime_estimator
from qnn.primitives.factory import make_estimator

algorithm_globals.random_seed = 42
num_inputs, num_samples = 2, 60
X = 2 * algorithm_globals.random.random([num_samples, num_inputs]) - 1
y01 = (np.sum(X, axis=1) >= 0).astype(int)
y = 2*y01 - 1
service = QiskitRuntimeService()
backend = service.backend("ibm_torino")  # set your backend
estimator_rt, session = make_runtime_estimator(service=service,backend_name=ackend.name, shots=4096, optimization_level=1, resilience_level=1)
try:
    qc = make_qnn_circuit(num_qubits=num_inputs)
    est = make_estimator("runtime", runtime_estimator=estimator_rt)
    qnn = make_estimator_qnn(qc, estimator=est)
    clf = make_classifier(qnn, optimizer=COBYLA(maxiter=40))
    clf.fit(X, y)
    print("train accuracy (hardware runtime):", clf.score(X, y))
finally:
    session.close()
