# %% Imports
import torch
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
torch.set_num_threads(30)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Keep the same theta for all tests
theta = torch.tensor(np.load("theta.npy"))

# %%

# Settings for the quantum circuit
N_WIRES  = 4
N_LAYERS = 4
N_SHOTS = 10000
PARALLEL_QUBITS = 30


# %% Initialize quantum devices
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
import os

qdevice_state_simulator = qml.device("default.qubit", wires=N_WIRES, shots=None)

qdevice_shots_simulator = qml.device("default.qubit", wires=N_WIRES, shots=N_SHOTS)

service = QiskitRuntimeService(channel="ibm_quantum")
real_backend = service.get_backend("ibm_brisbane")
aer_simulator = AerSimulator.from_backend(real_backend)
qdevice_noise_shots_simulator = qml.device("qiskit.aer", wires=N_WIRES, backend=aer_simulator, shots=N_SHOTS) 

IBMQ_API_TOKEN = os.getenv('IBMQ_API_TOKEN')
QiskitRuntimeService.save_account(channel="ibm_quantum", token=IBMQ_API_TOKEN, overwrite=True)
backend = service.least_busy(operational=True, simulator=False)
qdevice_real_hardware = qml.device("qiskit.remote", wires=PARALLEL_QUBITS, backend=backend, shots=N_SHOTS)

# %% Define the quantum circuit
def circuit(x, theta):
    # Embedding layer
    for i in range(N_WIRES):
        qml.RY(x, wires=i)

    # Variational layer
    for i in range(N_LAYERS):
        for j in range(N_WIRES):
            qml.RY(theta[i,j,0], wires=j)
            qml.RZ(theta[i,j,1], wires=j)

        for j in range(N_WIRES - 1):
            qml.CNOT(wires=[j, j + 1]) 
        
        
    # Cost function
    return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(N_WIRES)]))

def circuit_parallel(x, theta):
    for idx in range(x.shape[0]):
        current_index = idx*N_WIRES
        # Embedding layer
        for i in range(N_WIRES):
            qml.RY(x[idx], wires=current_index+i)
        # Variational layer
        for i in range(N_LAYERS):
            for j in range(N_WIRES):
                qml.RY(theta[i,j,0], wires=current_index+j)
                qml.RZ(theta[i,j,1], wires=current_index+j)
            for j in range(N_WIRES - 1):
                qml.CNOT(wires=[j, j + 1]) 
    # Cost function
    return [qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(idx*N_WIRES,idx*N_WIRES+N_WIRES)])) for idx in range(x.shape[0])]

qnode_state_simulator = qml.QNode(circuit, device=qdevice_state_simulator)
qnode_shots_simulator = qml.QNode(circuit, device=qdevice_shots_simulator)
qnode_noise_shots_simulator = qml.QNode(circuit, device=qdevice_noise_shots_simulator)
qnode_real_hardware = qml.QNode(circuit_parallel, device=qdevice_real_hardware)

def evaluate_circuit(x, theta, qnode):
    output = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        output[i] = qnode(x[i], theta)
    return output

def evaluate_circuit_parallel(x, theta, qnode, disp=True):
    chunk_size = min(PARALLEL_QUBITS // N_WIRES, x.shape[0])
    print(f"Chunk size: {chunk_size}")
    output = np.zeros(x.shape[0])
    
    for i in range((x.shape[0] + chunk_size - 1) // chunk_size):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, x.shape[0])
        print(f"Starting chunk {i+1} of {(x.shape[0] + chunk_size - 1) // chunk_size}")
        print(f"Indices: {start_idx} to {end_idx}")
        output[start_idx:end_idx] = qnode(x[start_idx:end_idx], theta)
        if disp: print(f"Chunk {i+1} of {(x.shape[0] + chunk_size - 1) // chunk_size} done")
    
    return output
# %%  Local Test
x = torch.linspace(0, 2*np.pi, 30)
x_dense = torch.linspace(0, 2*np.pi, 100)

# %%

u_state = evaluate_circuit(x_dense, theta, qnode_state_simulator)
u_shots = evaluate_circuit(x, theta, qnode_shots_simulator)
u_noise_shots = evaluate_circuit(x, theta, qnode_noise_shots_simulator)

# %% Real Hardware Test
u_real = evaluate_circuit_parallel(x, theta, qnode_real_hardware, disp=True)

# %%

plt.plot(x_dense, u_state, label="State simulator")
plt.scatter(x, u_shots, label="Shots simulator")
plt.scatter(x, u_noise_shots, label="Noise simulator")
plt.scatter(x, u_real, label="Real hardware")
plt.legend()
plt.show()
# %%
# save data as dictionary
data = {
    "x": x,
    "x_dense": x_dense,
    "u_state": u_state,
    "u_shots": u_shots,
    "u_noise_shots": u_noise_shots,
    "u_real": u_real,
    "layers": N_LAYERS,
    "wires": N_WIRES,
    "shots": N_SHOTS,
    "theta": theta.numpy()
}
# export data
np.save(f"real_hardware_noise_analysis_{N_LAYERS}layers_{N_WIRES}wires_{N_SHOTS}shots.npy", data)
