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
torch.set_num_threads(8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
## Constants
# FNN Basis Net
HIDDEN_LAYERS_FNN = 2
NEURONS_FNN = 9

# Domain Parameter
X_COLLOC_POINTS = 50
Y_COLLOC_POINTS = 25
BOUNDARY_SCALE = 10e1

# %%
##  Generate Domain

# Generate Collocation Points
x = torch.linspace(0.0, 2.0, X_COLLOC_POINTS)
y = torch.linspace(0.0, 1.0, Y_COLLOC_POINTS)
input_domain = torch.tensor(list(product(x, y)))

dir_boundary_mask = (input_domain[:, 0] == 0.0) | (input_domain[:, 0] == 2.0)
dir_boundary_colloc = input_domain[dir_boundary_mask]

# Neumann Boundary
neu_boundary_mask = (input_domain[:, 1] == 0.0) | (input_domain[:, 1] == 1.0)
neu_boundary_colloc = input_domain[neu_boundary_mask & ~dir_boundary_mask] 

# Combined Boundary Mask
boundary_mask = dir_boundary_mask | neu_boundary_mask

# Filter out boundary points from domain_colloc
interior_colloc = input_domain[~boundary_mask]

input_domain = input_domain.clone().detach().requires_grad_(True).to(device)
dir_boundary_colloc = dir_boundary_colloc.clone().detach().requires_grad_(True).to(device)
neu_boundary_colloc = neu_boundary_colloc.clone().detach().requires_grad_(True).to(device)
interior_colloc = interior_colloc.clone().detach().requires_grad_(True).to(device)

domain_bounds = torch.tensor([[0.0, 0.0], [2.0, 1.0]], device=device)

# # Plot domain
# plt.scatter(dir_boundary_colloc[:,0].detach().cpu(),dir_boundary_colloc[:,1].detach().cpu(), c="r", label="Dirichlet", s=10)
# plt.scatter(neu_boundary_colloc[:,0].detach().cpu(),neu_boundary_colloc[:,1].detach().cpu(), c="blue", label="Neumann", s=10)
# plt.scatter(interior_colloc[:,0].detach().cpu(),interior_colloc[:,1].detach().cpu(), c="black", label="Interior", s=10)
# plt.grid()
# plt.legend()
# plt.show()

# %%
## Create the Model

# Define QPINN
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
        return circuit_qnode(x_rescaled.T, basisNet(x_rescaled).T)
    else:
        return circuit_qnode(x_rescaled.T)


# %%
## Load the reference solution
base_dir = os.path.dirname(os.path.abspath(__file__))
u_interp = np.load(os.path.join(base_dir, "poisson_reference_solution.npy"), allow_pickle=True).item()

def reference_solution(data):
    output = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        output[i] = u_interp([data[i, 0], data[i, 1]]).squeeze()
    return output

reference_values = torch.tensor(reference_solution(input_domain.detach().cpu()), device=device)

# %%
## Define the problem
def source_term(x):
    return 10 * torch.exp(-((x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2) / 0.02)

def neu_boundary_term(x):
    return torch.sin(5 * x[:, 0])

def dir_loss_fnc():
    u_dir = model(dir_boundary_colloc)
    return torch.mean(u_dir**2)

def neu_loss_fnc():
    u_neu = model(neu_boundary_colloc)

    du_d = torch.autograd.grad(u_neu, neu_boundary_colloc, grad_outputs=torch.ones_like(u_neu), create_graph=True)[0]
    du_dy = du_d[:, 1]

    # Flip signs for first values, since they have to be outward facing relative to the domain
    du_dy[::2] *= -1.0

    g = neu_boundary_term(neu_boundary_colloc)
    return torch.mean((du_dy - g) ** 2)

def pde_loss_fnc():
    u = model(interior_colloc)
    f = source_term(interior_colloc)

    grad_out = torch.ones_like(u)
    du_d = torch.autograd.grad(u, interior_colloc, grad_outputs=grad_out, create_graph=True)[0]
    du_dx = du_d[:, 0]
    du_dy = du_d[:, 1]

    du_dxd = torch.autograd.grad(du_dx, interior_colloc, grad_outputs=grad_out, create_graph=True)[0]
    du_dxdx = du_dxd[:, 0]

    du_dyd = torch.autograd.grad(du_dy, interior_colloc, grad_outputs=grad_out, create_graph=True)[0]
    du_dydy = du_dyd[:, 1]

    pde_res = -du_dxdx - du_dydy - f
    return torch.mean(pde_res**2)


def loss_fnc():
    dir_loss = dir_loss_fnc()
    neu_loss = neu_loss_fnc()
    pde_loss = pde_loss_fnc()

    return BOUNDARY_SCALE * (dir_loss + neu_loss) + pde_loss

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

EMBEDDING_LIST = ["TOWER_CHEBYSHEV"]

data = np.zeros((5,4,2)) # layer, qubits, (loss, MSE_re)

for EMBEDDING in EMBEDDING_LIST:
    for k,N_LAYERS in enumerate([5]):
        for l,N_WIRES in enumerate([2,4]):
            print(f"Embedding: {EMBEDDING} \t Layers: {N_LAYERS} \t Qubits: {N_WIRES}")
            
            circuit_qnode = qml.QNode(circuit, device=qml.device("default.qubit", wires=N_WIRES), max_diff=2)
            theta = torch.rand(N_LAYERS, N_WIRES, 3, device=device, requires_grad=True)

            if EMBEDDING == "FNN_BASIS":
                basisNet = FNNBasisNet(HIDDEN_LAYERS_FNN, NEURONS_FNN).to(device)
                opt = torch.optim.LBFGS([theta, *basisNet.parameters()], line_search_fn="strong_wolfe")
            else:
                opt = torch.optim.LBFGS([theta], line_search_fn="strong_wolfe")


            previous_loss = float('inf')
            for i in range(100):
                opt.step(closure)
                print(f"Epoch {i}, Loss: {loss_fnc().item():.2E}", end="\r")

                if previous_loss == loss_fnc().item():
                    break
                previous_loss = loss_fnc().item()
                
            data[k,l,0] = loss_fnc().item()
            data[k,l,1] = compute_MSE_ref()         

                
            print(f"Final Loss: {loss_fnc().item():.2E} \t Iteration: {i} \t Embedding: {EMBEDDING} \t Layers: {N_LAYERS} \t Qubits: {N_WIRES} \t Iterations: {i} \t L2_Norm {compute_MSE_ref():.2E} \t L_Max_Norm {compute_lmax_norm():.2E}")


# %%



