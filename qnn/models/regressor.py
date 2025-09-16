
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
def make_regressor(qnn, optimizer, loss='squared_error', callback=None):
    return NeuralNetworkRegressor(neural_network=qnn, optimizer=optimizer, loss=loss, callback=callback)
