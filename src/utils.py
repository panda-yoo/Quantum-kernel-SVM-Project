# Importing Required Libraries


import pandas as pd
import numpy as np
from numpy.typing import NDArray

from tqdm import tqdm
from typing import Callable, Tuple, List

from qiskit.circuit.library import unitary_overlap
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector


def featureMap(circuit: QuantumCircuit, phi: Callable) -> Tuple[List[Parameter], QuantumCircuit]:
    """This function will Provide us with our Parmatrized 

    Args:
        circuit (QuantumCircuit): Pass a blank quantum circuit
        phi (Callable): can choose any function which depends on 2 features f(x,y) 

    Returns:
        Tuple[List[Parameter], QuantumCircuit]: it returns with 3 parameters in circuit ,and the modified Quantum circuit 
    """

    x0 = Parameter('x0')
    x1 = Parameter('x1')
    lam = Parameter('λ')

    params = [x0, x1, lam]

    #--------------------------------| PAULI ROTATION ALONG Y|--------------------------------

    circuit.ry(x0, 0)
    circuit.ry(x1, 1)

    #--------------------------------| INTERACTION BETWEEN 2 FEATURES WTIH PARAMETERIZED|--------------------------------
    #--------------------------------|  THETA ,THAT DEPENDS ON LAMDA,AND 2 FEATURES  |--------------------------------
    
    theta = lam * phi(x0, x1)

    circuit.cx(0, 1)                # CNOT gate for entanglement
    circuit.rz(theta, 1)                
    circuit.cx(0, 1)

    #--------------------------------| PAULI ROTATION ALONG Y|--------------------------------
    circuit.ry(x0, 0)
    circuit.ry(x1, 1)

    #--------------------------------| PAULI ROTATION ALONG Z|--------------------------------
    circuit.rz(theta, 0)
    circuit.rz(theta, 1)

    return params, circuit


def train_model(parameters:List[Parameter], base_qc:QuantumCircuit,
                train_kernel:NDArray, test_kernel:NDArray,
                X_train:NDArray, X_test:NDArray,
                lam:float)->Tuple[NDArray,NDArray]:
    """It uses the provided Quantum ciruit to calculate kernal matrices by assigning the normalized values from 
    the features to the gates in our parameterized circuit .And by assigning those values later it takes overlap/inner product of
    the circuits i.e U and U^dagger 

    Args:
        parameters (List[Parameter]): parameter we will use to modify gates 
        base_qc (QuantumCircuit): Modified Quantum circuit
        train_kernel (NDArray): numpy array to assign/fill values ,so further which can be used in SVM
        test_kernel (NDArray): numpy array used for testing the SVM Model
        X_train (NDArray): numpy array of training data from dataset with 2 features 
        X_test (NDArray): numpy array of test data from dataset with 2 features 
        lam (float): the parameter which can be used to modify theta -> phi  = lam*f(feature1,feature2)

    Returns:
        Tuple[NDArray,NDArray]: provides us training kernel and testing kernel 
    """
    
    
    train_size = X_train.shape[0]
    test_size  = X_test.shape[0]

    #--------------------------------| TRAIN KERNEL |--------------------------------
    for x1 in tqdm(range(train_size), desc=f'train λ={lam}'):
        for x2 in range(x1, train_size):

            u1 = base_qc.assign_parameters({
                parameters[0]: X_train[x1, 0],
                parameters[1]: X_train[x1, 1],
                parameters[2]: lam
            })

            u2 = base_qc.assign_parameters({
                parameters[0]: X_train[x2, 0],
                parameters[1]: X_train[x2, 1],
                parameters[2]: lam
            })

            ucir = unitary_overlap(u1, u2)

            psi = Statevector.from_instruction(ucir)
            prob_00 = np.abs(psi.data[0])**2

            train_kernel[x1, x2] = prob_00
            train_kernel[x2, x1] = prob_00

    #--------------------------------| TEST KERNEL |--------------------------------
    for x1 in tqdm(range(test_size), desc=f'test λ={lam}'):
        for x2 in range(train_size):

            u1 = base_qc.assign_parameters({
                parameters[0]: X_test[x1, 0],
                parameters[1]: X_test[x1, 1],
                parameters[2]: lam
            })

            u2 = base_qc.assign_parameters({
                parameters[0]: X_train[x2, 0],
                parameters[1]: X_train[x2, 1],
                parameters[2]: lam
            })

            ucir = unitary_overlap(u1, u2)

            psi = Statevector.from_instruction(ucir)
            prob_00 = np.abs(psi.data[0])**2

            test_kernel[x1, x2] = prob_00

    return train_kernel, test_kernel

