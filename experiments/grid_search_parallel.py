
import numpy as np
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.optimizers import COBYLA
from qnn.circuits.qnn_circuit import make_qnn_circuit
from qnn.networks.estimator_qnn import make_estimator_qnn
from qnn.models.classifier import make_classifier
from qnn.primitives.factory import make_estimator
from qnn.training.parallel import parallel_cv_fit_score

algorithm_globals.random_seed = 42
num_inputs, num_samples = 2, 120
X = 2 * algorithm_globals.random.random([num_samples, num_inputs]) - 1
y01 = (np.sum(X, axis=1) >= 0).astype(int)
y = 2*y01 - 1
estimator = make_estimator("statevector")

def build_model(params):
    qc = make_qnn_circuit(num_qubits=num_inputs, ansatz_kwargs={"reps": params["reps"]})
    qnn = make_estimator_qnn(qc, estimator=estimator)
    opt = COBYLA(maxiter=params["maxiter"])
    return make_classifier(qnn, optimizer=opt)

param_grid = [{"reps": 1, "maxiter": 30},{"reps": 2, "maxiter": 30},{"reps": 1, "maxiter": 60},{"reps": 2, "maxiter": 60}]
results = parallel_cv_fit_score(build_model, X, y, params_list=param_grid, k=5, n_jobs=-1)
print("CV results (best first):")
for params, score in results:
    print(params, "->", score)
