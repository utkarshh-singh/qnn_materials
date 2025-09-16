
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
def make_classifier(qnn, optimizer, loss='cross_entropy', callback=None):
    return NeuralNetworkClassifier(neural_network=qnn, optimizer=optimizer, loss=loss, callback=callback)
