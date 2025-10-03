# %%
import pennylane as qml
import torch
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

N_QUBITS = 4
N_LAYERS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dev = qml.device("default.qubit", wires=N_QUBITS)

# Set default tensor type to float32
torch.set_default_tensor_type(torch.DoubleTensor)

class FNNBasisNet(torch.nn.Module):
    def __init__(self, n_hidden_layers, branch_width):
        super().__init__()
        self.n_hidden_layers = n_hidden_layers
        self.branch_width = branch_width
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(1, branch_width))
        for i in range(n_hidden_layers-1):
            self.layers.append(torch.nn.Linear(branch_width, branch_width))
        self.layers.append(torch.nn.Linear(branch_width, N_QUBITS))
    
    def forward(self, x):
        for i in range(self.n_hidden_layers):
            x = torch.tanh(self.layers[i](x))
        x = self.layers[self.n_hidden_layers](x)
        return x

def circuit(x, basis, theta):
    # Embedding
    for i in range(N_QUBITS):
        qml.RY(basis[i] * x, wires=i)
    
    # Variational ansatz
    for i in range(N_LAYERS):
        for j in range(N_QUBITS):
            qml.RX(theta[i,j,0], wires=j)
            qml.RY(theta[i,j,1], wires=j)
            qml.RZ(theta[i,j,2], wires=j)
    
        for j in range(N_WIRES - 1):
            qml.CNOT(wires=[j, j + 1])
 
    ## Cost Function
    return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(N_QUBITS)]))

qnode = qml.QNode(circuit, dev)

def model(x, basisNet, theta):
    # Rescale input to [-0.95, 0.95]       
    x_rescaled = 0.95 * 2*x - 0.95
    
    return qnode(x_rescaled.T, basisNet(x_rescaled.unsqueeze(1)).T, theta)

def loss_diff_fnc(basisNet, theta):
    x = torch.linspace(0.0, 1.0, 100, requires_grad=True, device=device)
    u = model(x, basisNet, theta) 
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    res = du_dx - (4*u - 6*u**2 + torch.sin(50*x) + u*torch.cos(25*x) - 0.5)
    return torch.mean(res**2)

def loss_boundary_fnc(basisNet, theta):
    x = torch.linspace(0.0, 1.0, 100, requires_grad=True, device=device)
    u_0 = model(torch.zeros_like(x), basisNet, theta)
    return torch.mean((u_0 - 0.75)**2)

def loss_fnc(basisNet, theta):
    loss_diff     = loss_diff_fnc(basisNet, theta)
    loss_boundary = loss_boundary_fnc(basisNet, theta)
    return 10E1*loss_boundary + loss_diff        

def run_lbfgs_opt(n=200):
    basisNet = FNNBasisNet(1, 10).to(device)
    theta = torch.rand(N_LAYERS, N_QUBITS, 3, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([theta, *basisNet.parameters()], line_search_fn="strong_wolfe")
    
    def closure():
        opt.zero_grad() 
        loss = loss_fnc(basisNet, theta)
        loss.backward()
        return loss

    loss_history_lbfgs = []
    for i in range(n):
        opt.step(closure)
        current_loss = loss_fnc(basisNet, theta).item()
        loss_history_lbfgs.append(current_loss)
        print(f"Iteration: {i}, Loss: {current_loss}")

    return loss_history_lbfgs


def run_adam_opt(n=5000):
    basisNet = FNNBasisNet(1, 5).to(device)
    theta = torch.rand(N_LAYERS, N_QUBITS, 3, device=device, requires_grad=True)
    opt = torch.optim.Adam([theta, *basisNet.parameters()], lr=0.01)

    loss_history_adam = []
    for i in range(n):
        opt.zero_grad()
        loss = loss_fnc(basisNet, theta)
        loss.backward()
        opt.step()
        loss_history_adam.append(loss.item())
        print(f"Iteration: {i}, Loss: {loss.item()}")

    return loss_history_adam

def run_cobyla_opt(n=5000):
    basisNet = FNNBasisNet(1, 5).to(device)
    theta = np.random.rand(N_LAYERS, N_QUBITS, 3)
    
    def flatten_params():
        flattened_theta = theta.flatten()
        flattened_basis = np.concatenate([p.data.cpu().numpy().flatten() for p in basisNet.parameters()])
        return np.concatenate((flattened_theta, flattened_basis))

    def unflatten_params(params):
        nonlocal theta
        n_theta = N_LAYERS * N_QUBITS * 3
        theta = params[:n_theta].reshape((N_LAYERS, N_QUBITS, 3))
        
        idx = n_theta
        for p in basisNet.parameters():
            p_size = p.numel()
            p.data = torch.tensor(params[idx:idx+p_size].reshape(p.shape), device=device)
            idx += p_size
            
    def objective(params):
        unflatten_params(params)
        return loss_fnc(basisNet, theta).item()
    
    initial_params = flatten_params()
    
    loss_history_COBYLA = []
    
    iteration = 0
    def callback(params):
        nonlocal iteration
        iteration += 1
        loss = objective(params)
        loss_history_COBYLA.append(loss)
        print(f"Iteration {iteration}, Loss: {loss}")
        
    result = minimize(objective, initial_params, method='COBYLA', options={'maxiter': n-1}, callback=callback)
    
    return loss_history_COBYLA
    
lbfgs_data = []
cobyla_data = []
adam_data = []

# %%
# Compute each optimizer 10 times and take the average


for i in range(3):
    loss_history_COBYLA = run_cobyla_opt(n=10000)
    cobyla_data.append(loss_history_COBYLA)

    loss_history_adam = run_adam_opt(n=5000)
    adam_data.append(loss_history_adam)

    loss_history_lbfgs = run_lbfgs_opt(n=100)
    lbfgs_data.append(loss_history_lbfgs)

# %%

lbfgs_mean = np.mean(lbfgs_data, axis=0)
cobyla_mean = np.mean(cobyla_data, axis=0)
adam_mean = np.mean(adam_data, axis=0)

# Plot the loss histories with low alpha and the mean
plt.figure(figsize=(10, 5))

for i in range(len(cobyla_data)):
    # plt.plot(lbfgs_data[i], alpha=0.4, color="red")
    plt.plot(cobyla_data[i], alpha=0.4, color="blue")
    plt.plot(adam_data[i], alpha=0.4, color="green")

plt.plot(lbfgs_mean, label="L-BFGS", color="red")
plt.plot(cobyla_mean, label="COBYLA", color="blue")
plt.plot(adam_mean, label="Adam", color="green")
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.grid()
plt.show()

# %%

print("idx lbfgs_1 lbfgs_2 lbfgs_3 lbfgs_mean")
for i in range(len(lbfgs_data[0])):
    print(f"{i} {lbfgs_data[0][i]} {lbfgs_data[1][i]} {lbfgs_data[2][i]} {lbfgs_mean[i]}")
# %%
print("idx cobyla_1 cobyla_2 cobyla_3 cobyla_mean")
for i in range(len(cobyla_data[0])):
    print(f"{i} {cobyla_data[0][i]} {cobyla_data[1][i]} {cobyla_data[2][i]} {cobyla_mean[i]}")
# %%
print("idx adam_1 adam_2 adam_3 adam_mean")
for i in range(len(adam_data[0])):
    print(f"{i} {adam_data[0][i]} {adam_data[1][i]} {adam_data[2][i]} {adam_mean[i]}")
# %%
