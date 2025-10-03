# %%

import numpy as np
import scipy
import scipy.interpolate
import matplotlib.pyplot as plt

# load data

data_1_layers = np.load("real_hardware_noise_analysis_1layers_4wires_10000shots.npy", allow_pickle=True).item()
data_4_layers = np.load("real_hardware_noise_analysis_4layers_4wires_10000shots.npy", allow_pickle=True).item()
data_8_layers = np.load("real_hardware_noise_analysis_8layers_4wires_10000shots.npy", allow_pickle=True).item()
data_12_layers = np.load("real_hardware_noise_analysis_12layers_4wires_10000shots.npy", allow_pickle=True).item()
data_24_layers = np.load("real_hardware_noise_analysis_24layers_4wires_10000shots.npy", allow_pickle=True).item()
data_48_layers = np.load("real_hardware_noise_analysis_48layers_4wires_10000shots.npy", allow_pickle=True).item()

data_1batch = np.load("real_hardware_noise_analysis_1layers_4wires_10000shots_120parallelqubits.npy", allow_pickle=True).item()
data_2batch = np.load("real_hardware_noise_analysis_1layers_4wires_10000shots_60parallelqubits.npy", allow_pickle=True).item()
data_3batch = np.load("real_hardware_noise_analysis_1layers_4wires_10000shots_40parallelqubits.npy", allow_pickle=True).item()

data_4_layers_1batch = np.load("real_hardware_noise_analysis_4layers_4wires_10000shots_1batches.npy", allow_pickle=True).item()
data_4_layers_2batch = np.load("real_hardware_noise_analysis_4layers_4wires_10000shots_2batches.npy", allow_pickle=True).item()
data_4_layers_3batch = np.load("real_hardware_noise_analysis_4layers_4wires_10000shots_3batches.npy", allow_pickle=True).item()


data_1shot = np.load("real_hardware_noise_analysis_4layers_4wires_1shots.npy", allow_pickle=True).item()
data_10shots = np.load("real_hardware_noise_analysis_4layers_4wires_10shots.npy", allow_pickle=True).item()
data_100shots = np.load("real_hardware_noise_analysis_4layers_4wires_100shots.npy", allow_pickle=True).item()
data_1000shots = np.load("real_hardware_noise_analysis_4layers_4wires_1000shots.npy", allow_pickle=True).item()
data_10000shots = np.load("real_hardware_noise_analysis_4layers_4wires_10000shots.npy", allow_pickle=True).item()
data_100000shots = np.load("real_hardware_noise_analysis_4layers_4wires_100000shots.npy", allow_pickle=True).item()
# %%


error_shots = []
error_noise = []
error_real = []

# Load data
data_layers = [data_1_layers, data_4_layers, data_8_layers] #, data_12_layers, data_24_layers, data_48_layers]
for data in data_layers:
    x_dense = data["x_dense"]
    u_state = data["u_state"]
    ground_truth = scipy.interpolate.interp1d(x_dense, u_state, kind="linear")

    error_shots.append(np.mean((data["u_shots"] - ground_truth(data["x"]))**2))
    error_noise.append(np.mean((data["u_noise_shots"] - ground_truth(data["x"]))**2))
    error_real.append(np.mean((data["u_real"] - ground_truth(data["x"]))**2))

layers = [1,4,8 ]#, 12 ], 24, 48]

plt.scatter(layers, error_shots, label="shots")
plt.scatter(layers, error_noise, label="noise")
plt.scatter(layers, error_real, label="real")
plt.xlabel("Number of layers")
plt.ylabel("Mean squared error")
plt.yscale("log")
plt.legend()
plt.show()

print("layers error_shots error_noise error_real")
for layer in layers:
    print(layer, error_shots[layers.index(layer)], error_noise[layers.index(layer)], error_real[layers.index(layer)])

# %%

error_shots = []
error_noise = []
error_real = []

data_shots = [data_1shot, data_10shots, data_100shots, data_1000shots, data_10000shots, data_100000shots]

for data in data_shots:
    x_dense = data["x_dense"]
    u_state = data["u_state"]
    ground_truth = scipy.interpolate.interp1d(x_dense, u_state, kind="linear")

    error_shots.append(np.mean((data["u_shots"] - ground_truth(data["x"]))**2))
    error_noise.append(np.mean((data["u_noise_shots"] - ground_truth(data["x"]))**2))
    error_real.append(np.mean((data["u_real"] - ground_truth(data["x"]))**2))
    
shots = [1,10,100,1000,10000,100000]

plt.scatter(shots, error_shots, label="shots")
plt.scatter(shots, error_noise, label="noise")
plt.scatter(shots, error_real, label="real")
plt.xlabel("Number of shots")
plt.ylabel("Mean squared error")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()

print("shots error_shots error_noise error_real")
for shot in shots:
    print(shot, error_shots[shots.index(shot)], error_noise[shots.index(shot)], error_real[shots.index(shot)])


# %%

data_batches = [data_1batch, data_2batch, data_3batch]

error_noise = []
error_real = []

for data in data_batches:
    x_dense = data["x_dense"]
    u_state = data["u_state"]
    ground_truth = scipy.interpolate.interp1d(x_dense, u_state, kind="linear")

    error_noise.append(np.mean((data["u_noise_shots"] - ground_truth(data["x"]))**2))
    error_real.append(np.mean((data["u_real"] - ground_truth(data["x"]))**2))
    
batches = [1,2,3]

utilization = [120/127, 60/127, 40/127]

plt.scatter(utilization, error_noise, label="noise")
plt.scatter(utilization, error_real, label="real")
plt.xlabel("Utilization")
plt.ylabel("Mean squared error")
plt.yscale("log")
plt.legend()
plt.show()

print("utilization error_noise error_real")
for ut in utilization:
    print(ut, error_noise[utilization.index(ut)], error_real[utilization.index(ut)])
# %%

data_batches = [data_4_layers_1batch, data_4_layers_2batch, data_4_layers_3batch]

error_noise = []
error_real = []

for data in data_batches:
    x_dense = data["x_dense"]
    u_state = data["u_state"]
    ground_truth = scipy.interpolate.interp1d(x_dense, u_state, kind="linear")

    error_noise.append(np.mean((data["u_noise_shots"] - ground_truth(data["x"]))**2))
    error_real.append(np.mean((data["u_real"] - ground_truth(data["x"]))**2))

batches = [1,2,3]

utilization = [120/127, 60/127, 40/127]

plt.scatter(utilization, error_noise, label="noise")
plt.scatter(utilization, error_real, label="real")
plt.xlabel("Utilization")
plt.ylabel("Mean squared error")
plt.yscale("log")
plt.legend()
plt.show()

# %%
