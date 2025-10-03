# %%
import numpy as np

data_1batch = np.load("real_hardware_noise_analysis_1layers_4wires_10000shots_120parallelqubits.npy", allow_pickle=True).item()
data_2batch = np.load("real_hardware_noise_analysis_1layers_4wires_10000shots_60parallelqubits.npy", allow_pickle=True).item()
data_3batch = np.load("real_hardware_noise_analysis_1layers_4wires_10000shots_40parallelqubits.npy", allow_pickle=True).item()
# %%
print("x u_state")
for i in range(len(data_1batch["x_dense"])):
    print(data_1batch["x_dense"].numpy()[i], data_1batch["u_state"][i])
# %%
print("x u_shots_1batch u_shots_2batch u_shots_3batch u_noise_shots_1batch u_noise_shots_2batch u_noise_shots_3batch u_real_1batch u_real_2batch u_real_3batch")
for i in range(len(data_1batch["x"])):
    print(data_1batch["x"].numpy()[i], data_1batch["u_shots"][i], data_2batch["u_shots"][i], data_3batch["u_shots"][i], data_1batch["u_noise_shots"][i], data_2batch["u_noise_shots"][i], data_3batch["u_noise_shots"][i], data_1batch["u_real"][i], data_2batch["u_real"][i], data_3batch["u_real"][i])

# %%
