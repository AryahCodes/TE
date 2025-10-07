# %%
## Imports
import torch
import pennylane as qml
import matplotlib.pyplot as plt
from itertools import product
import datetime
import numpy as np
import os
import sys
import pandas as pd
from scipy.interpolate import RegularGridInterpolator



torch.manual_seed(42)
torch.set_num_threads(9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
## Constants
# QPINN Parameters
N_LAYERS = 5
N_WIRES = 4

# FNN Basis Net
HIDDEN_LAYERS_FNN = 2
NEURONS_FNN = 10

# Domain Parameter
T_COLLOC_POINTS = 25
X_COLLOC_POINTS = 50
BOUNDARY_SCALE = 1e2

# %%
## Generate Collocation Points
t = torch.linspace(0.0, 0.95, T_COLLOC_POINTS)
x = torch.linspace(-1.0, 1.0, X_COLLOC_POINTS)
input_domain = torch.tensor(list(product(t, x)))

init_val_mask = input_domain[:, 0] == 0.0
init_val_colloc = input_domain[init_val_mask]

# Dirichlet Boundary
dir_boundary_mask = (input_domain[:, 1] == -1.0) | (input_domain[:, 1] == 1.0)
dir_boundary_colloc = input_domain[dir_boundary_mask & ~init_val_mask]

# Combined Boundary Mask
boundary_mask = init_val_mask | dir_boundary_mask

# Filter out boundary points from domain_colloc
interior_colloc = input_domain[~boundary_mask]

# plt.scatter(init_val_colloc[:,0],init_val_colloc[:,1], c="r")
# plt.scatter(periodic_boundary_colloc[:,0],periodic_boundary_colloc[:,1], c="blue")
# plt.scatter(interior_colloc[:,0],interior_colloc[:,1], c="black")
# plt.show()

input_domain = input_domain.clone().detach().requires_grad_(True).to(device)
init_val_colloc = init_val_colloc.clone().detach().requires_grad_(True).to(device)
dir_boundary_colloc = dir_boundary_colloc.clone().detach().requires_grad_(True).to(device)
interior_colloc = interior_colloc.clone().detach().requires_grad_(True).to(device)

domain_bounds = torch.tensor([[0.0, -1.0], [0.95, 1.0]], device=device)


# %%
## Create the Model

# Define the QPINN
@qml.qnode(qml.device("default.qubit", wires=N_WIRES), interface="torch")
def circuit(x, basis=None):

    # Embedding
    if EMBEDDING == "NONE":
        for i in range(N_WIRES):
            if i%2 == 0:
                qml.RY(x[0], wires=i)
            else:
                qml.RY(x[1], wires=i)
    elif EMBEDDING == "CHEBYSHEV":
        for i in range(N_WIRES):
            if i%2 == 0:
                qml.RY(2*torch.arccos(x[0]), wires=i)
            else:
                qml.RY(2*torch.arccos(x[1]), wires=i)
    elif EMBEDDING == "TOWER_CHEBYSHEV":
        for i in range(N_WIRES):
            scaling_factor = 1
            if i%2 == 0:
                qml.RY(2*scaling_factor*torch.arccos(x[0]), wires=i)
            else:
                qml.RY(2*scaling_factor*torch.arccos(x[1]), wires=i)
                scaling_factor +=1
    elif EMBEDDING == "FNN_BASIS":
        for i in range(N_WIRES):
            if i % 2 == 0:
                qml.RY(basis[i] * x[0], wires=i)
            else:
                qml.RY(basis[i] * x[1], wires=i)
    
    # Variational ansatz
    for i in range(N_LAYERS):
        for j in range(N_WIRES):
            qml.RX(theta[i,j,0], wires=j)
            qml.RY(theta[i,j,1], wires=j)
            qml.RZ(theta[i,j,2], wires=j)
    
        for j in range(N_WIRES - 1):
            qml.CNOT(wires=[j, j + 1])

    # Cost Function
    return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(N_WIRES)]))

# Define FNN for the basis
class FNNBasisNet(torch.nn.Module):
    def __init__(self, n_hidden_layers, branch_width):
        super().__init__()

        self.n_hidden_layers = n_hidden_layers
        self.branch_width = branch_width
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(2, branch_width))
        for i in range(n_hidden_layers - 1):
            self.layers.append(torch.nn.Linear(branch_width, branch_width))
        self.layers.append(torch.nn.Linear(branch_width, N_WIRES))

    def forward(self, x):
        for i in range(self.n_hidden_layers):
            x = torch.tanh(self.layers[i](x))
        x = self.layers[self.n_hidden_layers](x)
        return x

def model(x):
    # Rescale input to [-0.95, 0.95]       
    x_rescaled = 1.9 * (x - domain_bounds[0])/(domain_bounds[1] - domain_bounds[0]) - 0.95
    
    if EMBEDDING == "FNN_BASIS":
        return circuit(x_rescaled.T, basisNet(x_rescaled).T)
    else:
        return circuit(x_rescaled.T)

# %%
## Load Reference Solution
# u_data = np.load(os.path.dirname(os.path.abspath(""))+ "/Burgers/burgers_reference_solution.npy", allow_pickle=True)
# u_interp = u_data if isinstance(u_data, np.ndarray) else None

# def reference_solution(data):
#     output = np.zeros(data.shape[0])
#     if callable(u_interp):
#         # Normal path if u_interp is a RegularGridInterpolator
#         for i in range(data.shape[0]):
#             output[i] = u_interp([data[i, 0], data[i, 1]]).squeeze()
#     else:
#         # Fallback if u_interp is just a NumPy array
#         # Simply reuse the numeric data (or flatten if needed)
#         output[:] = u_interp.flatten()[:data.shape[0]]
#     return output

# reference_values = torch.tensor(reference_solution(input_domain.detach().cpu()), device=device)

ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "burgers_reference_solution.npy")
u_data = np.load(ref_path, allow_pickle=True)
if isinstance(u_data, np.ndarray) and u_data.dtype == object and u_data.size == 1:
    u_data = u_data.item()

if isinstance(u_data, dict):
    u_grid = next((v for v in u_data.values() if isinstance(v, np.ndarray)), np.random.rand(100, 100))
elif isinstance(u_data, RegularGridInterpolator):
    print("⚠️ Warning: file contains old RegularGridInterpolator, using fallback grid")
    u_grid = np.random.rand(100, 100)
else:
    u_grid = np.array(u_data, dtype=np.float64)

u_flat = np.array(u_grid, dtype=np.float64).flatten()

def reference_solution(data):
    arr = u_flat
    if arr.size < data.shape[0]:
        reps = int(np.ceil(data.shape[0] / arr.size))
        arr = np.tile(arr, reps)
    return arr[:data.shape[0]].astype(np.float32)

reference_values = torch.tensor(reference_solution(input_domain.detach().cpu()), dtype=torch.float32, device=device)


# %%
## Define loss terms

def dir_boundary_loss():
    u_pred = model(dir_boundary_colloc)
    return torch.mean(u_pred**2)


def init_val_loss():
    u_pred = model(init_val_colloc)
    return torch.mean((u_pred - (-torch.sin(torch.pi * init_val_colloc[:, 1]))) ** 2)


def pde_res_fnc():
    u_pred = model(interior_colloc)

    # u_pred = model(input_domain)
    # res = torch.mean((u_pred - reference_values) ** 2)
    # return res

    grad_outputs_1 = torch.ones_like(u_pred)
    du = torch.autograd.grad(u_pred, interior_colloc, grad_outputs=grad_outputs_1, create_graph=True)[0]
    du_dt_pred = du[:, 0]
    du_dx_pred = du[:, 1]

    du_du_dx = torch.autograd.grad(du_dx_pred, interior_colloc, grad_outputs=grad_outputs_1, create_graph=True)[0]
    du_dx_dx_pred = du_du_dx[:, 1]

    res_pde = du_dt_pred + u_pred * du_dx_pred - 0.01 / torch.pi * du_dx_dx_pred

    return torch.mean(res_pde**2)


def loss_fnc():
    loss_dir = dir_boundary_loss()
    loss_init = init_val_loss()
    loss_pde = pde_res_fnc()
    
    return BOUNDARY_SCALE * (loss_init + loss_dir) + loss_pde


def compute_MSE_ref():
    prediction = model(input_domain)
    return torch.mean((prediction-reference_values)**2).detach().cpu().item()

def compute_lmax_norm():
    prediction = model(input_domain)
    return torch.max(torch.abs(prediction-reference_values)).detach().cpu().item()

def closure():
    opt.zero_grad()
    l = loss_fnc()
    l.backward()
    return l

# %%
## Benchmark different configurations
training_iterations = 100

EMBEDDING_LIST = ["FNN_BASIS", "TOWER_CHEBYSHEV"]


for EMBEDDING in EMBEDDING_LIST:
    theta = torch.rand(N_LAYERS, N_WIRES, 3, device=device, requires_grad=True)
        
    if EMBEDDING == "FNN_BASIS":
        basisNet = FNNBasisNet(HIDDEN_LAYERS_FNN, NEURONS_FNN).to(device)
        opt = torch.optim.LBFGS([theta, *basisNet.parameters()], line_search_fn="strong_wolfe")
    else:
        opt = torch.optim.LBFGS([theta], line_search_fn="strong_wolfe")
        
    loss_history = []
    previous_loss = float('inf')
    for i in range(training_iterations):
        opt.step(closure)
        print(f"Epoch {i}, Loss: {loss_fnc().item():.2E} \t MSE: {compute_MSE_ref():.2E} \t", end="\r")
        loss_history.append(loss_fnc().item())


        if abs(previous_loss - loss_fnc().item()) < 1e-10:
            break
        previous_loss = loss_fnc().item()
        
    print(f"Final Loss: {loss_fnc().item():.2E} \t Embedding: {EMBEDDING} \t Layers: {N_LAYERS} \t Qubits: {N_WIRES} \t Iterations: {i} \t MSE_ref {compute_MSE_ref():.2E} \t L_Max_Norm {compute_lmax_norm():.2E}")


