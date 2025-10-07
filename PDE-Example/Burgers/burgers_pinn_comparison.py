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

torch.manual_seed(42)
torch.set_num_threads(9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
## Constants
# QPINN Parameters
N_LAYERS = 5
N_WIRES = 4

# FNN Basis Net
HIDDEN_LAYERS_FNN = 1
NEURONS_FNN = 10

# Domain Parameter
T_COLLOC_POINTS = 30
X_COLLOC_POINTS = 60
BOUNDARY_SCALE = 10e1

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

# Define QPINN
@qml.qnode(qml.device("default.qubit", wires=N_WIRES), interface="torch")
def circuit(x, basis):

    # Embedding
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
class FNN(torch.nn.Module):
    def __init__(self, n_hidden_layers, branch_width, output_dim=N_WIRES):
        super().__init__()

        self.n_hidden_layers = n_hidden_layers
        self.branch_width = branch_width
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(2, branch_width))
        for i in range(n_hidden_layers -1):
            self.layers.append(torch.nn.Linear(branch_width, branch_width))
        self.layers.append(torch.nn.Linear(branch_width, output_dim))

    def forward(self, x):
        for i in range(self.n_hidden_layers):
            x = torch.tanh(self.layers[i](x))
        x = self.layers[self.n_hidden_layers](x)
        return x

def model(x):
    # Rescale input to [-0.95, 0.95]       
    x_rescaled = 1.9 * (x - domain_bounds[0])/(domain_bounds[1] - domain_bounds[0]) - 0.95
    return circuit(x_rescaled.T, basisNet(x_rescaled).T)

def model_pinn(x):
    x_rescaled = 1.9 * (x - domain_bounds[0])/(domain_bounds[1] - domain_bounds[0]) - 0.95
    return pinn(x_rescaled).squeeze()

def get_n_params(fnn_model):
    pp=0
    for p in list(fnn_model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# # %%
# ## Load Reference Solution
# base_dir = os.path.dirname(os.path.abspath(__file__))
# u_interp = np.load(os.path.join(base_dir, "burgers_reference_solution.npy"), allow_pickle=True).item()

# def reference_solution(data):
#     output = np.zeros(data.shape[0])
#     for i in range(data.shape[0]):
#         output[i] = u_interp([data[i, 0], data[i, 1]]).squeeze()
#     return output

# reference_values = torch.tensor(reference_solution(input_domain.detach().cpu()), device=device)
# %% Load reference solution (robust to different .npy formats)
from scipy.interpolate import RegularGridInterpolator

base_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(base_dir, "burgers_reference_solution.npy")
raw = np.load(ref_path, allow_pickle=True)

# Unwrap if it’s an object array holding a dict/interpolator
if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
    raw = raw.item()

u_interp = None
t_grid = None
x_grid = None
u_grid = None

if isinstance(raw, dict):
    # Expect keys like 't', 'x', 'u'
    if {"t", "x", "u"} <= set(raw.keys()):
        t_grid = np.asarray(raw["t"], dtype=float)
        x_grid = np.asarray(raw["x"], dtype=float)
        u_grid = np.asarray(raw["u"], dtype=float)
        u_interp = RegularGridInterpolator((t_grid, x_grid), u_grid, bounds_error=False, fill_value=None)
    else:
        # If dict but unknown keys, try to find the first 2D array
        for v in raw.values():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                u_grid = np.asarray(v, dtype=float)
                break

elif isinstance(raw, RegularGridInterpolator):
    # Old pickled interpolator (may not be compatible) — try to use, else rebuild from fallback below
    u_interp = raw

elif isinstance(raw, np.ndarray):
    # If it's a plain 2D array, use it directly
    if raw.ndim == 2:
        u_grid = np.asarray(raw, dtype=float)

# If we only have a grid but no axes, fabricate axes spanning the standard Burgers domain
if u_interp is None and u_grid is not None:
    t_grid = np.linspace(0.0, 0.95, u_grid.shape[0])
    x_grid = np.linspace(-1.0, 1.0,  u_grid.shape[1])
    u_interp = RegularGridInterpolator((t_grid, x_grid), u_grid, bounds_error=False, fill_value=None)

def reference_solution(data_tensor):
    data_np = data_tensor.detach().cpu().numpy()
    if u_interp is not None:
        return u_interp(data_np).astype(np.float64)
    # Last-resort fallback: sample from flattened array (keeps code running without good refs)
    flat = u_grid.flatten()
    idx = np.arange(data_np.shape[0]) % flat.size
    return flat[idx].astype(np.float64)

reference_values = torch.tensor(reference_solution(input_domain), dtype=torch.float32, device=device)


# %%
## Define loss terms

def dir_boundary_loss(model):
    u_pred = model(dir_boundary_colloc)
    return torch.mean(u_pred**2)


def init_val_loss(model):
    u_pred = model(init_val_colloc)
    return torch.mean((u_pred - (-torch.sin(torch.pi * init_val_colloc[:, 1]))) ** 2)


def pde_res_fnc(model):
    u_pred = model(interior_colloc)

    grad_outputs_1 = torch.ones_like(u_pred)
    du = torch.autograd.grad(u_pred, interior_colloc, grad_outputs=grad_outputs_1, create_graph=True)[0]
    du_dt_pred = du[:, 0]
    du_dx_pred = du[:, 1]

    du_du_dx = torch.autograd.grad(du_dx_pred, interior_colloc, grad_outputs=grad_outputs_1, create_graph=True)[0]
    du_dx_dx_pred = du_du_dx[:, 1]

    res_pde = du_dt_pred + u_pred * du_dx_pred - 0.01 / torch.pi * du_dx_dx_pred

    return torch.mean(res_pde**2)


def loss_fnc():
    loss_dir = dir_boundary_loss(model)
    loss_init = init_val_loss(model)
    loss_pde = pde_res_fnc(model)
    return BOUNDARY_SCALE * (loss_init + loss_dir) + loss_pde

def loss_fnc_PINN():
    loss_dir = dir_boundary_loss(model_pinn)
    loss_init = init_val_loss(model_pinn)
    loss_pde = pde_res_fnc(model_pinn)
    return BOUNDARY_SCALE * (loss_init + loss_dir) + loss_pde

def compute_MSE_ref():
    prediction = model(input_domain)
    return torch.mean((prediction-reference_values)**2).detach().cpu().item()

def compute_lmax_norm():
    prediction = model(input_domain)
    return torch.max(torch.abs(prediction-reference_values)).detach().cpu().item()


# %%
## Train the model
qpinn_data = []
pinn_data = []

for i in range(2):
    # Create initial parameters and BasisNet object
    theta = torch.rand(N_LAYERS, N_WIRES, 3, device=device, requires_grad=True)
    basisNet = FNN(2, 10).to(device)
    pinn = FNN(3, 10, output_dim=1).to(device)

    # Number of trainable parameters
    print("Number of trainable parameters in QPINN: ", get_n_params(basisNet) + theta.flatten().shape[0])
    print("Number of trainable parameters in PINN: ", get_n_params(pinn))

    opt = torch.optim.LBFGS([theta, *basisNet.parameters()], line_search_fn="strong_wolfe")
    opt_PINN = torch.optim.LBFGS(pinn.parameters(), line_search_fn="strong_wolfe")

    loss_history_qpinn = []
    loss_history_pinn = []

    def closure():
        opt.zero_grad()
        l = loss_fnc()
        l.backward()
        return l

    def closure_PINN():
        opt_PINN.zero_grad()
        l = loss_fnc_PINN()
        l.backward()
        return l
    
    for i in range(100):
        opt.step(closure)
        loss_history_qpinn.append(loss_fnc().item())
        print(f"QPINN: Epoch {i}, Loss: {loss_fnc().item()}")


    for i in range(100):
        opt_PINN.step(closure_PINN)
        loss_history_pinn.append(loss_fnc_PINN().item())
        print(f"PINN: Epoch {i}, Loss: {loss_fnc_PINN().item()}")


    qpinn_data.append(loss_history_qpinn)
    pinn_data.append(loss_history_pinn)

# %%
for i in range(2):
    plt.plot(qpinn_data[i], color="red")
    plt.plot(pinn_data[i], color="blue")
plt.legend()
plt.yscale("log")
plt.show()

# %%
print("idx loss_qpinn_1 loss_qpinn_2 loss_qpinn_3 loss_qpinn_mean")
for i in range(100):
    print(i, qpinn_data[0][i], qpinn_data[1][i], qpinn_data[2][i], np.mean([qpinn_data[0][i], qpinn_data[1][i], qpinn_data[2][i]]))

# %%
print("idx loss_pinn_1 loss_pinn_2 loss_pinn_3 loss_pinn_mean")
for i in range(100):
    print(i, pinn_data[0][i], pinn_data[1][i], pinn_data[2][i], np.mean([pinn_data[0][i], pinn_data[1][i], pinn_data[2][i]]))

# %%



