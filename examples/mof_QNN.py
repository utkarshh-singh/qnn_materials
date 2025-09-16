#
# 01/09/25 in Betim-MG
#

from numpy import arange
import numpy as np
from matplotlib import pyplot

# QML
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#########################

file_name='regression_data_60.txt'
f = open(file_name, 'r')
n_features=4
n_data = sum(1 for _ in open(file_name, 'r'))
print("n_data:", n_data)
data_list = [[0.0 for i in range(n_features)] for j in range(n_data)]
z_list = []

counter01 = 0
for iLine in f:
    tmp01_f = []
    tmp01_f = iLine.split()
    for iFeature in range(len(tmp01_f)):
         if iFeature < n_features:
             data_list[counter01][iFeature] = float(tmp01_f[iFeature])
         if iFeature == n_features:
              z_list.append(float(tmp01_f[iFeature]))

    counter01 = counter01 + 1

print("data_list:", data_list, len(data_list))
print("z_list:", z_list, len(z_list))

from sklearn.preprocessing import StandardScaler

X_train_non_scaled, X_test_non_scaled, y_train, y_test = train_test_split(data_list, z_list, test_size=0.20, random_state=42)

## Scaling ##
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_non_scaled)
X_test = scaler.transform(X_test_non_scaled)
## End scaling ##

##### ???? #####
print("X_train:", X_train)
print("type(X_train):", type(X_train))

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

### NN BEGIN ##
from squlearn import Executor
from squlearn.encoding_circuit import (YZ_CX_EncodingCircuit)
from squlearn.observables import IsingHamiltonian
from squlearn.optimizers import Adam
from squlearn.qnn import QNNRegressor, SquaredLoss

num_qubits=len(X_train[0])
print("num_qubits:", num_qubits)
num_layers=1

pqc = YZ_CX_EncodingCircuit(num_qubits=num_qubits, num_features=num_qubits, num_layers=num_layers)
print (pqc.draw("mpl").savefig('fig_QNN' + '_layers_' + str(num_layers) + '.png'))

obs = IsingHamiltonian(num_qubits=num_qubits)
qnn = QNNRegressor(pqc, obs, Executor(), SquaredLoss(), Adam())

print("X_train:", X_train)
print("y_train:", y_train)

print("model QNN:", qnn)

qnn.fit(X_train, y_train)

## QNN END ##

predictions_train = qnn.predict(X_train)
print("\npredictions_train:", predictions_train)
mae_train =  mean_absolute_error(y_train, predictions_train)
print("mae_train:", mae_train)
mse_train =  mean_squared_error(y_train, predictions_train)
print("mse_train:", mse_train)
r2_train = r2_score(y_train, predictions_train)
print("r2_train:", r2_train)

predictions_test = qnn.predict(X_test)
print("\npredictions_test:", predictions_test)
mae_test =  mean_absolute_error(y_test, predictions_test)
print("mae_test:", mae_test)
mse_test =  mean_squared_error(y_test, predictions_test)
print("mse_test:", mse_test)
r2_test = r2_score(y_test, predictions_test)
print("r2_test:", r2_test)

##### ???? #####

print("R2 of predicted 'all data' from data_list: {:.4f}".format( qnn.score(data_list, z_list)) )
print("R2 of predicted 'train' from X_train: {:.4f}".format( qnn.score(X_train, y_train)) )
print("R2 of predicted 'test' from Y_test: {:.4f}".format( qnn.score(X_test, y_test)) )

#### plots ####
left_plot_range = min(z_list)
right_plot_range = max(z_list)
fig, (ax0) = plt.subplots(nrows=1, sharex=True)
y_grid = np.arange(left_plot_range - left_plot_range*0.05, right_plot_range + right_plot_range*0.05, 0.1)
fit = np.polyfit(y_train, y_train, 1)
fit_fn = np.poly1d(fit)
ax0.plot(fit_fn(y_grid), fit_fn(y_grid), '--y', label="y_train", lw=2)

ax0.errorbar(y_train, predictions_train, fmt='o', markersize=6, markeredgecolor='k', label="y_train_predicted", elinewidth=1.4, capthick=0.4, capsize=4)
ax0.plot(y_test, predictions_test, 'o', markersize=8, markeredgecolor='k', label="y_test_predicted", color="red")
ax0.legend(loc='upper left')
ax0.set_xlabel("Observed", fontsize=12)
ax0.set_ylabel("Predicted", fontsize=12)

iOptIteration=0
plt.title("QNN-sQUlearn: \n" +
" Train:" + " MAE: " + str(round( mae_train , 4)) + ", RMSE: " + str( round( mse_train , 4)) + "\n" +
"Test:" + " Av.MAE: " + str(round( mae_test , 4)) + ", RMSE: " + str( round( mse_test , 4))
)

plt.savefig("QNN_sQUlearn_plot_ego_"  + str(iOptIteration) + "_QNN_" + "layers" + str(num_layers) + '.png', dpi=100)
# END GRAPH 01
#### END PLOT ####

# show the plot
pyplot.show()
