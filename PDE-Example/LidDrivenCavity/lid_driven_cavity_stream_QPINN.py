# %%
import pennylane as qml
import torch
from torch import nn
import matplotlib.pyplot as plt
from itertools import product
import datetime
import numpy as np
import os
import sys


torch.manual_seed(42)
torch.set_num_threads(30)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# QNet Parameter
N_WIRES = 6
N_LAYERS = 5

# Domain Parameter
X_COLLOC_POINTS = 35
Y_COLLOC_POINTS = 35
BOUNDARY_SCALE = 5e1

# FNN Basis Net
NEURONS_FNN = 8

# PDE Parameter
REYNOLDS_NUMBER = 10.0

## Generate Collocation Points
domain_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device)
x = torch.linspace(0.0, 1.0, X_COLLOC_POINTS)
y = torch.linspace(0.0, 1.0, Y_COLLOC_POINTS)
input_domain = torch.tensor(list(product(x, y)))

lid_boundary_mask = input_domain[:, 1] == 1.0
lid_boundary_colloc = input_domain[lid_boundary_mask]

no_slip_boundary_mask = (
      (input_domain[:, 0] == 0.0)
    | (input_domain[:, 0] == 1.0)
    | (input_domain[:, 1] == 0.0) & (input_domain[:, 1] != 1.0)
)
no_slip_boundary_colloc = input_domain[no_slip_boundary_mask]

# Combined Boundary Mask
boundary_mask = no_slip_boundary_mask | lid_boundary_mask

# Filter out boundary points from domain_colloc
interior_colloc = input_domain[~boundary_mask]

# plt.scatter(no_slip_boundary_colloc[:,0],no_slip_boundary_colloc[:,1], c="r")
# plt.scatter(lid_boundary_colloc[:,0],lid_boundary_colloc[:,1], c="blue")
# plt.scatter(interior_colloc[:,0],interior_colloc[:,1], c="black")
# plt.show()

input_domain = input_domain.clone().detach().requires_grad_(True).to(device)
no_slip_boundary_colloc = no_slip_boundary_colloc.clone().detach().requires_grad_(True).to(device)
lid_boundary_colloc = lid_boundary_colloc.clone().detach().requires_grad_(True).to(device)
interior_colloc = interior_colloc.clone().detach().requires_grad_(True).to(device)


## Load reference solution
interpolator_p =  np.load("ref_interpolator_p.npy", allow_pickle=True).item()
interpolator_u_mag = np.load("ref_interpolator_u_mag.npy", allow_pickle=True).item()

p_ref = interpolator_p(input_domain.cpu().detach().numpy()).T
u_mag_ref = interpolator_u_mag(input_domain.cpu().detach().numpy()).T

def compute_l2_error():
    p, u, v = model(input_domain)
    u_mag = torch.sqrt(u**2 + v**2)
    
    error_p = torch.norm(p - torch.tensor(p_ref, device=device), p=2)
    error_u_mag = torch.norm(u_mag - torch.tensor(u_mag_ref, device=device), p=2)
    return error_p, error_u_mag


## Define model
class ModelFFNBasis(torch.nn.Module):
    def __init__(self, domain_bounds, basis_width, n_wires, n_layers):
        super().__init__()

        self.basis_width = basis_width
        self.n_wires = n_wires
        self.n_layers = n_layers

        self.domain_bounds = domain_bounds

        self.sequential_stack = torch.nn.Sequential(
            nn.Linear(2, self.basis_width),
            nn.Tanh(),
            nn.Linear(self.basis_width, self.basis_width),
            nn.Tanh(),
            nn.Linear(self.basis_width, self.basis_width),
            nn.Tanh(),
            nn.Linear(self.basis_width, self.n_wires),
        )

        self.weights = nn.Parameter(torch.randn(self.n_layers, self.n_wires, 3, requires_grad=True))

        self.qnode = qml.QNode(
            self.circuit,
            qml.device("default.qubit", wires=self.n_wires),
            interface="torch",
            diff_method="best",
            max_diff=3,
        )

    def forward(self, x):
        x_rescaled = 1.9 * (x - domain_bounds[0])/(domain_bounds[1] - domain_bounds[0]) - 0.95
        all_basis = self.sequential_stack(x_rescaled)
        return self.qnode(x_rescaled.T, all_basis.T)

    def circuit(self, x, basis):
        # Embedding Ansatz
        for i in range(self.n_wires):
            if i % 2 == 0:
                qml.RY(basis[i] * x[0], wires=i)
            else:
                qml.RY(basis[i] * x[1], wires=i)
        
        # Variational ansatz
        for i in range(self.n_layers):
            for j in range(self.n_wires):
                qml.RX(self.weights[i,j,0], wires=j)
                qml.RY(self.weights[i,j,1], wires=j)
                qml.RZ(self.weights[i,j,2], wires=j)
        
            for j in range(N_WIRES - 1):
                qml.CNOT(wires=[j, j + 1])

        return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(self.n_wires)]))



model_p = ModelFFNBasis(domain_bounds, NEURONS_FNN, N_WIRES, N_LAYERS).to(device)
model_stream = ModelFFNBasis(domain_bounds, NEURONS_FNN, N_WIRES, N_LAYERS).to(device)

def model(x):
    output_p = 5*model_p(x)
    output_stream = 5*model_stream(x)

    d_stream = torch.autograd.grad(output_stream, x, grad_outputs=torch.ones_like(output_stream), create_graph=True)[0]
    output_u = d_stream[:, 1]
    output_v = -1.0 * d_stream[:, 0]

    return output_p, output_u, output_v

## Define loss terms
def no_slip_loss_fnc():
    _, u, v = model(no_slip_boundary_colloc)
    return torch.mean(u**2 + v**2)


def lid_driven_loss_fnc():
    _, u, v = model(lid_boundary_colloc)
    return torch.mean(v**2 + (u - torch.ones_like(u)) ** 2)


def pde_loss_fnc():
    p, u, v = model(interior_colloc)

    grad_out = torch.ones_like(u)
    dp_d = torch.autograd.grad(p, interior_colloc, grad_outputs=grad_out, create_graph=True)[0]
    dp_dx = dp_d[:, 0]
    dp_dy = dp_d[:, 1]

    du_d = torch.autograd.grad(u, interior_colloc, grad_outputs=grad_out, create_graph=True)[0]
    du_dx = du_d[:, 0]
    du_dy = du_d[:, 1]

    dv_d = torch.autograd.grad(v, interior_colloc, grad_outputs=grad_out, create_graph=True)[0]
    dv_dx = dv_d[:, 0]
    dv_dy = dv_d[:, 1]

    du_dxd = torch.autograd.grad(du_dx, interior_colloc, grad_outputs=grad_out, create_graph=True)[0]
    du_dxdx = du_dxd[:, 0]
    du_dyd = torch.autograd.grad(du_dy, interior_colloc, grad_outputs=grad_out, create_graph=True)[0]
    du_dydy = du_dyd[:, 1]

    dv_dxd = torch.autograd.grad(dv_dx, interior_colloc, grad_outputs=grad_out, create_graph=True)[0]
    dv_dxdx = dv_dxd[:, 0]
    dv_dyd = torch.autograd.grad(dv_dy, interior_colloc, grad_outputs=grad_out, create_graph=True)[0]
    dv_dydy = dv_dyd[:, 1]

    res_mom_x = u * du_dx + v * du_dy + dp_dx - 1 / REYNOLDS_NUMBER * (du_dxdx + du_dydy)
    res_mom_y = u * dv_dx + v * dv_dy + dp_dy - 1 / REYNOLDS_NUMBER * (dv_dxdx + dv_dydy)

    return torch.mean(res_mom_x**2 + res_mom_y**2)


def reference_loss_fnc():
    p, u, v = model(torch.tensor([[0.0, 0.0]], device=device, requires_grad=True))
    return torch.mean(p**2)


def loss_fnc():
    no_slip_loss = no_slip_loss_fnc()
    lid_driven_loss = lid_driven_loss_fnc()
    pde_loss = pde_loss_fnc()

    p_ref_loss = reference_loss_fnc()
    return BOUNDARY_SCALE * (no_slip_loss + lid_driven_loss+ p_ref_loss) + pde_loss


## Define optimizer
opt = torch.optim.LBFGS([*model_p.parameters(), *model_stream.parameters()], line_search_fn="strong_wolfe")


def closure():
    opt.zero_grad()
    l = loss_fnc()
    l.backward()
    return l

loss_hist = []
loss_reference_p = []
loss_reference_u_mag = []

# %%
## Trainings loop
for i in range(150):
    start = datetime.datetime.now()
    opt.step(closure)

    loss_hist.append(loss_fnc().detach().cpu().numpy())
    error_p, error_u_mag = compute_l2_error()
    loss_reference_p.append(error_p.detach().cpu().numpy())
    loss_reference_u_mag.append(error_u_mag.detach().cpu().numpy())

    print(f"Iteration: {i} | Loss: {loss_hist[-1]:.8e} | l2 Error P: {error_p:.8e} | l2 Error U Mag: {error_u_mag:.8e}")
    print(f"\t Iteration duration: {(datetime.datetime.now()-start).total_seconds()}")
# %%
## Plot result

fig = plt.figure(figsize=(10, 5), layout="constrained")
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax2 = fig.add_subplot(gs[:, 2])
fig.suptitle(f"LDC Testcase NS + Stream Function", fontsize=18)

# Set titles
ax0.set_title("Pressure")
ax1.set_title("Velocity Magnitude")
ax3.set_title("U Velocity")
ax4.set_title("V Velocity")

# Set aspect ratio
ax0.set_aspect("equal", "box")
ax1.set_aspect("equal", "box")
ax3.set_aspect("equal", "box")
ax4.set_aspect("equal", "box")

# Generate prediction
p, u, v = model(input_domain)
X, Y = torch.meshgrid(x, y)

# Plot the prediction
cs0 = ax0.contourf(
    X,
    Y,
    p.reshape(X_COLLOC_POINTS, Y_COLLOC_POINTS).detach().cpu(),
    400,
    cmap="Spectral",
)
cb0 = plt.colorbar(cs0, ax=ax0)
cs1 = ax1.contourf(
    X,
    Y,
    torch.sqrt(
        u.reshape(X_COLLOC_POINTS, Y_COLLOC_POINTS).detach().cpu() ** 2
        + v.reshape(X_COLLOC_POINTS, Y_COLLOC_POINTS).detach().cpu() ** 2
    ),
    400,
    cmap="plasma",
    )
cb1 = plt.colorbar(cs1, ax=ax1)
cs3 = ax3.contourf(
    X,
    Y,
    u.reshape(X_COLLOC_POINTS, Y_COLLOC_POINTS).detach().cpu(),
    400,
    cmap="plasma",
)
cb3 = plt.colorbar(cs3, ax=ax3)
cs4 = ax4.contourf(
    X,
    Y,
    v.reshape(X_COLLOC_POINTS, Y_COLLOC_POINTS).detach().cpu(),
    400,
    cmap="plasma",
)
cb4 = plt.colorbar(cs4, ax=ax4)

# Plot the difference
# cs1 = ax1.contourf(X, Y, np.abs(diff_reference()), 200, cmap="plasma")
# cb1 = plt.colorbar(cs1, ax=ax1)

# Plot the loss history
ax2.plot(loss_hist, label="Loss QPINN")
ax2.plot(loss_reference_p, label="Loss Reference P")
ax2.plot(loss_reference_u_mag, label="Loss Reference U Mag")
ax2.set_yscale("log")
ax2.grid()
ax2.legend(loc="upper right")

plt.show()

# %%

x_red = torch.linspace(0.0, 1.0, 50, requires_grad=True)
y_red = torch.linspace(0.0, 1.0, 50, requires_grad=True)
input_domain_red = torch.tensor(list(product(x_red, y_red)), device=device, requires_grad=True)

p_ref = interpolator_p(input_domain_red.cpu().detach().numpy()).T
u_mag_ref = interpolator_u_mag(input_domain_red.cpu().detach().numpy()).T



p,u, v = model(input_domain_red)
print("x y u v u_mag p u_mag_ref p_ref u_mag_diff p_diff")
for i in range(input_domain_red.shape[0]):
    print(f"{input_domain_red[i,0].item()} {input_domain_red[i,1].item()} {u[i].item()} {v[i].item()} {torch.sqrt(u[i]**2 + v[i]**2).item()} {p[i].item()} {u_mag_ref[i].item()} {p_ref[i].item()} {torch.sqrt((u_mag_ref[i] - torch.sqrt(u[i]**2 + v[i]**2))**2).item()} {torch.sqrt((p_ref[i] - p[i])**2).item()}")

# %%
print("idx loss_hist u_mag_loss p_loss")
for i in range(len(loss_hist)):
    print(f"{i} {loss_hist[i]} {loss_reference_u_mag[i]} {loss_reference_p[i]}")

# %%

if __name__ == "__main__":
    import sys
    # If you pass '--plot-only', skip the training loop
    if "--plot-only" in sys.argv:
        print("Plot-only mode: skipping training loop.")
        # Skip directly to the plotting section
        # You can load any previous log file or saved state here if desired
    else:
        # Normal behavior
        for i in range(150):
            start = datetime.datetime.now()
            opt.step(closure)
            loss_hist.append(loss_fnc().detach().cpu().numpy())
            error_p, error_u_mag = compute_l2_error()
            loss_reference_p.append(error_p.detach().cpu().numpy())
            loss_reference_u_mag.append(error_u_mag.detach().cpu().numpy())
            print(f"Iteration: {i} | Loss: {loss_hist[-1]:.8e} | l2 Error P: {error_p:.8e} | l2 Error U Mag: {error_u_mag:.8e}")
            print(f"\t Iteration duration: {(datetime.datetime.now()-start).total_seconds()}")
