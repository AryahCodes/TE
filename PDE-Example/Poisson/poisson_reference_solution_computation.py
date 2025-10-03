# %%  
import importlib.util

if importlib.util.find_spec("petsc4py") is not None:
    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
    from petsc4py.PETSc import ScalarType
else:
    print("This demo requires petsc4py.")
    exit(0)


import numpy as np
import matplotlib.pyplot as plt
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner
from petsc4py import PETSc
from  scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import numpy as np

# Function to solve the problem for a given mesh resolution
def solve_poisson(n):
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (2.0, 1.0)),
        n=n,
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(msh, ("Lagrange", 1))

    facets = mesh.locate_entities_boundary(
        msh,
        dim=(msh.topology.dim - 1),
        marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0),
    )

    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc = fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dofs, V=V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    g = ufl.sin(5 * x[0])
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx + inner(g, v) * ds

    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    return msh, V, uh

msh, V, uh = solve_poisson((200,100))


x = msh.geometry.x[:, 0]
y = msh.geometry.x[:, 1]
u_vals = uh.x.array.real

interp = LinearNDInterpolator(np.column_stack((x, y)), u_vals)

x = np.linspace(0, 2, 200)
y = np.linspace(0, 1, 100)

X, Y = np.meshgrid(x, y)
Z = interp(np.column_stack((X.flatten(), Y.flatten()))).reshape(X.shape)

plt.contourf(X,Y, Z, cmap="coolwarm", levels=100)
plt.title("Poisson solution")
plt.xlabel("x") 
plt.ylabel("y")
plt.colorbar()
plt.show()


# Export interpoloation
np.save("poisson_reference_solution.npy", interp)
