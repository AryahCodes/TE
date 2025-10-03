# %%
import torch
import matplotlib.pyplot as plt
from itertools import product
import datetime
import numpy as np
import torch.nn as nn
import pennylane as qml
import os


torch.manual_seed(42)
torch.set_num_threads(30)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# QNet Parameter
N_WIRES = 1

qdevice = qml.device("default.qubit", wires=N_WIRES)

# FNN Basis Net
BRANCH_WIDTH = 5
N_LAYERS_FNN = 1

x = torch.linspace(0.0, torch.pi, 100, requires_grad=True, device=device)

class FNNBasisNet(torch.nn.Module):
    def __init__(self, n_hidden_layers, branch_width):
        super().__init__()

        self.n_hidden_layers = n_hidden_layers
        self.branch_width = branch_width
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(1, branch_width))
        for i in range(n_hidden_layers - 1):
            self.layers.append(torch.nn.Linear(branch_width, branch_width))
        self.layers.append(torch.nn.Linear(branch_width, N_WIRES))
    
    def forward(self, x):
        for i in range(self.n_hidden_layers):
            x = torch.tanh(self.layers[i](x))
        x = self.layers[self.n_hidden_layers](x)
        return x

def circuit(x):

    for i in range(N_LAYERS):
        for j in range(N_WIRES):
            qml.RX(theta[i,j,0] + x, wires=j)
            qml.RY(theta[i,j,1], wires=j)
            qml.RZ(theta[i,j,2], wires=j)

    return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(N_WIRES)]))

qnode = qml.QNode(circuit, qdevice)

def eval_circuit(x):
    x_rescaled = 2.0*x/np.pi - 1.0
    return qnode(x_rescaled)

def circuit_basis(x, basis):

    for i in range(N_LAYERS):
        for j in range(N_WIRES):
            qml.RX(basis[0]*x, wires=j)
            
            qml.RX(theta[i,j,0], wires=j)
            qml.RY(theta[i,j,1], wires=j)
            qml.RZ(theta[i,j,2], wires=j)

    return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(N_WIRES)]))

qnode_basis = qml.QNode(circuit_basis, qdevice)

def eval_circuit_basis(x):
    x_rescaled = 2.0*x/np.pi - 1.0
    return qnode_basis(x_rescaled.T, basisNet(x_rescaled.unsqueeze(1)).T)

analytical_solution = torch.exp(-x)*torch.sin(10*x)

def compute_MSE_reference():
    if FNN_BASIS:
        u = eval_circuit_basis(x)
    else :
        u = eval_circuit(x)
        
    return torch.mean((u - analytical_solution)**2)


def loss_fnc():
    if FNN_BASIS:
        u = eval_circuit_basis(x)
    else :
        u = eval_circuit(x)
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    res_pde = du_dx - (torch.exp(-x)*(10*torch.cos(10*x) - torch.sin(10*x)))
    return torch.mean(res_pde**2) + 10*torch.mean((u[0])**2)

def closure():
    opt.zero_grad()
    loss_val = loss_fnc()
    loss_val.backward()
    return loss_val

# %%
# Iterate over different N_LAYERS and compute the loss

loss_values_basis = []
mse_ref_values_basis = []
loss_values = []
mse_ref_values = []

N_LAYERS_values = [5, 10, 25, 50, 75, 100]
# %%
for N_LAYERS in N_LAYERS_values:
    print(f"Running for N_LAYERS = {N_LAYERS}")

    run_basis_loss = []
    run_basis_mse_ref = []
    run_loss = []
    run_mse_ref = []

    for run in range(5):
        print(f"Run {run}")

        FNN_BASIS = False
        theta  = torch.tensor(2*torch.pi*torch.rand(N_LAYERS, N_WIRES, 3),requires_grad=True, device=device)
        opt = torch.optim.LBFGS([theta], line_search_fn="strong_wolfe")
        
        previous_loss = float('inf')  # Initialize previous loss to infinity
        for i in range(500):
            opt.step(closure)
            current_loss = loss_fnc().item()
            print(f"Iteration {i}: \t Loss = {current_loss}",end='\r')
            
            # Check if the change in loss is below the threshold
            if previous_loss == current_loss:
                print(f"Iteration {i}: Loss = {current_loss}")
                print(f"Stopping early at iteration {i} due to small change in loss.")
                break
            
            previous_loss = current_loss

        run_loss.append(current_loss)
        run_mse_ref.append(compute_MSE_reference().item())

        FNN_BASIS = True
        theta  = torch.tensor(2*torch.pi*torch.rand(N_LAYERS, N_WIRES, 3),requires_grad=True, device=device)
        basisNet = FNNBasisNet(N_LAYERS_FNN, BRANCH_WIDTH).to(device)
        opt = torch.optim.LBFGS([theta, *basisNet.parameters()], line_search_fn="strong_wolfe")

        previous_loss = float('inf')  # Initialize previous loss to infinity
        for i in range(500):
            opt.step(closure)
            current_loss = loss_fnc().item()
            print(f"Iteration {i}: \t Loss = {current_loss}",end='\r')
            
            # Check if the change in loss is below the threshold
            if previous_loss == current_loss:
                print(f"Iteration {i}: Loss = {current_loss}")
                print(f"Stopping early at iteration {i} due to small change in loss.")
                break
            previous_loss = current_loss
        
        run_basis_loss.append(current_loss)
        run_basis_mse_ref.append(compute_MSE_reference().item())
    
    loss_values.append(run_loss)
    mse_ref_values.append(run_mse_ref)
    loss_values_basis.append(run_basis_loss)
    mse_ref_values_basis.append(run_basis_mse_ref)
# %%
import numpy as np
# Plot the mean values of each run
fig, ax = plt.subplots(1,1, figsize=(10,6))
ax.plot(N_LAYERS_values, np.mean(loss_values, axis=1), label="No Embedding", marker="o", c="r")
ax.plot(N_LAYERS_values, np.mean(loss_values_basis, axis=1), label="With Embedding", marker="o", c="b")

# add the min values
ax.plot(N_LAYERS_values, np.min(loss_values, axis=1), label="No Embedding (min)", marker="o", c="r", linestyle="--")
ax.plot(N_LAYERS_values, np.min(loss_values_basis, axis=1), label="With Embedding (min)", marker="o", c="b", linestyle="--")



ax.set_xlabel("N_LAYERS")
ax.set_ylabel("Loss")
ax.set_yscale("log")
ax.set_title(r"Loss vs N_LAYERS, $ \frac{du}{dx} = \exp^{-x}(10\cos(10x) - \sin(10x))$")
plt.grid()
ax.legend()
plt.show()

# %%
# Save all values of all runs
import pandas as pd

df = pd.DataFrame({
    "N_LAYERS": np.repeat(N_LAYERS_values, 5),
    "loss_values": np.array(loss_values).flatten(),
    "mse_ref_values": np.array(mse_ref_values).flatten(),
    "loss_values_basis": np.array(loss_values_basis).flatten(),
    "mse_ref_values_basis": np.array(mse_ref_values_basis).flatten(),
})


# df.to_csv("single_qubit_simple_example.csv", index=False)
# Print data frame as csv
print(df.to_csv(index=False))

# %%
print("N_LAYER mean_loss mean_loss_basis min_loss min_loss_basis mean_mse_ref mean_mse_ref_basis")
for i in range(len(N_LAYERS_values)):
    print(f"{N_LAYERS_values[i]} {np.mean(loss_values[i]):.4e} {np.mean(loss_values_basis[i]):.4e} {np.min(loss_values[i]):.4e} {np.min(loss_values_basis[i]):.4e} {np.mean(mse_ref_values[i]):.4e} {np.mean(mse_ref_values_basis[i]):.4e}")


# %%
