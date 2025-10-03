# %%
import numpy as np

data_1_layers = np.load("real_hardware_noise_analysis_1layers_4wires_10000shots.npy", allow_pickle=True).item()
data_4_layers = np.load("real_hardware_noise_analysis_4layers_4wires_10000shots.npy", allow_pickle=True).item()
data_8_layers = np.load("real_hardware_noise_analysis_8layers_4wires_10000shots.npy", allow_pickle=True).item()



# %%
# Print state solutions
print("x u_state_1layers u_state_4layers u_state_8layers")
for i in range(len(data_1_layers["x_dense"])):
    print(data_1_layers["x_dense"].numpy()[i], data_1_layers["u_state"][i] , data_4_layers["u_state"][i], data_8_layers["u_state"][i])

    

# %%

# Print different solutions
print("x u_shots_1layers u_shots_4layers u_shots_8layers u_noise_shots_1layers u_noise_shots_4layers u_noise_shots_8layers u_real_1layers u_real_4layers u_real_8layers")
for i in range(len(data_1_layers["x"])):
    print(data_1_layers["x"].numpy()[i], data_1_layers["u_shots"][i], data_4_layers["u_shots"][i], data_8_layers["u_shots"][i], data_1_layers["u_noise_shots"][i], data_4_layers["u_noise_shots"][i], data_8_layers["u_noise_shots"][i], data_1_layers["u_real"][i], data_4_layers["u_real"][i], data_8_layers["u_real"][i])

# %%
