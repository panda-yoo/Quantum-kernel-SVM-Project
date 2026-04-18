
import numpy as np
from numpy.typing import NDArray

from matplotlib.axes import Axes
import os

from typing import List

from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector


#  --------------------------| plot_decision_boundary_quantum |--------------------------

def plot_decision_boundary_quantum(
    model: SVC,
    base_qc :QuantumCircuit,
    parameters:List[Parameter],
    norm_X_train:NDArray,
    y_train:NDArray,
    ax:Axes,
    title:str ,
    lam:float=0.06,
    h:float=0.1   # coarser grid = faster
):
    # ---- grid ----
    x_min, x_max = norm_X_train[:, 0].min() - 0.5, norm_X_train[:, 0].max() + 0.5
    y_min, y_max = norm_X_train[:, 1].min() - 0.5, norm_X_train[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # ---- ONLY SUPPORT VECTORS ----
    support_indices = model.support_
    support_vectors = norm_X_train[support_indices]
    n_grid = grid_points.shape[0]
    n_sv   = support_vectors.shape[0]

    grid_kernel = np.zeros((n_grid, n_sv))

    # ---- precompute support vector states ----
    sv_states = []
    for j in range(n_sv):
        u2 = base_qc.assign_parameters({
            parameters[0]: support_vectors[j, 0],
            parameters[1]: support_vectors[j, 1],
            parameters[2]: lam
        })
        sv_states.append(Statevector.from_instruction(u2))

    # ---- compute kernel ----
    for i in range(n_grid):
        u1 = base_qc.assign_parameters({
            parameters[0]: grid_points[i, 0],
            parameters[1]: grid_points[i, 1],
            parameters[2]: lam
        })
        psi1 = Statevector.from_instruction(u1)

        for j in range(n_sv):
            overlap = np.abs(psi1.data.conj() @ sv_states[j].data) ** 2
            grid_kernel[i, j] = overlap

    print("n_support:", len(model.support_))
    print("dual_coef shape:", model.dual_coef_.shape)
    # ---- decision function manually ----
    dual_coef = model.dual_coef_[0]   # shape (n_sv,)
    intercept = model.intercept_[0]

    decision = grid_kernel @ dual_coef + intercept

    Z = decision.reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    ax.scatter(norm_X_train[:, 0], norm_X_train[:, 1], c=y_train, edgecolors='k')

    ax.set_title(f"Decision Boundary \n(λ = {title})")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")

#  --------------------------| plot_decision_boundary_classical |--------------------------
def plot_decision_boundary_classical(model:SVC, X_train:NDArray, y_train:NDArray, ax:Axes, title:str, h:float=0.02):

    # ---- grid ----
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    # ---- decision function ----
    Z = model.decision_function(grid)
    Z = Z.reshape(xx.shape)

    # ---- plot ----
    ax.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')

    ax.set_title(title)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")


def folder(folder_name:str):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name