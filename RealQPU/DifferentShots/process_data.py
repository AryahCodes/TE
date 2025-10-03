#%%
import numpy as np

data_1shot = np.load("real_hardware_noise_analysis_4layers_4wires_1shots.npy", allow_pickle=True).item()
data_10shots = np.load("real_hardware_noise_analysis_4layers_4wires_10shots.npy", allow_pickle=True).item()
data_100shots = np.load("real_hardware_noise_analysis_4layers_4wires_100shots.npy", allow_pickle=True).item()
data_1000shots = np.load("real_hardware_noise_analysis_4layers_4wires_1000shots.npy", allow_pickle=True).item()
data_10000shots = np.load("real_hardware_noise_analysis_4layers_4wires_10000shots.npy", allow_pickle=True).item()
data_100000shots = np.load("real_hardware_noise_analysis_4layers_4wires_100000shots.npy", allow_pickle=True).item()
# %%
print("x u_state")
for i in range(len(data_1shot["x_dense"])):
    print(data_1shot["x_dense"].numpy()[i], data_1shot["u_state"][i])

# %%
print("x u_shots_1shot u_shots_10shots u_shots_100shots u_shots_1000shots u_shots_10000shots u_shots_100000shots u_noise_shots_1shot u_noise_shots_10shots u_noise_shots_100shots u_noise_shots_1000shots u_noise_shots_10000shots u_noise_shots_100000shots u_real_1shot u_real_10shots u_real_100shots u_real_1000shots u_real_10000shots u_real_100000shots")
for i in range(len(data_1shot["x"])):
    print(data_1shot["x"].numpy()[i], data_1shot["u_shots"][i], data_10shots["u_shots"][i], data_100shots["u_shots"][i], data_1000shots["u_shots"][i], data_10000shots["u_shots"][i], data_100000shots["u_shots"][i], data_1shot["u_noise_shots"][i], data_10shots["u_noise_shots"][i], data_100shots["u_noise_shots"][i], data_1000shots["u_noise_shots"][i], data_10000shots["u_noise_shots"][i], data_100000shots["u_noise_shots"][i], data_1shot["u_real"][i], data_10shots["u_real"][i], data_100shots["u_real"][i], data_1000shots["u_real"][i], data_10000shots["u_real"][i], data_100000shots["u_real"][i])
# %%
