# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:11:13 2024

@author: katerina
"""

import torch.nn as nn
import torch
from qiskit.primitives import Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.circuit.library import ZZFeatureMap
from sklearn.decomposition import KernelPCA



# define the NN architecture
class PQAE(nn.Module):
    def __init__(self,input_dim,n_qubits,n_pca):
        super(PQAE, self).__init__()

        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_pca = n_pca

        ## encoder ##
        self.input = nn.Linear(self.input_dim, self.n_qubits, dtype=torch.float64)

        ## decoder ##
        self.decoder = nn.Linear(self.n_pca,self.input_dim, dtype=torch.float64)


        # Define the qubit states as column vectors
        self.q0 = torch.tensor([[1], [0]], dtype=torch.cfloat)
        self.q1 = torch.tensor([[0], [1]], dtype=torch.cfloat)

        # Define the Pauli matrices
        self.sigma1 = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64)
        self.sigma2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        self.sigma3 = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        self.sigma4 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

        self.hadamard_gate = 1 / torch.sqrt(torch.tensor(2) ) * torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64)

        # Define the CNOT gate
        self.cnot_gate = torch.tensor([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 0, 1],
                                       [0, 0, 1, 0]], dtype=torch.complex64)



        # Collect the Pauli matrices in a list
        self.generators = [ self.sigma2, self.sigma3, self.sigma4 ]



    def forward(self, x):
        # define feedforward behavior
        x = self.input(x)    # Here the data are reduced from 64 to 4, which is the number of qubits

        sampler = Sampler()     # Initialize a sampler which samples from the quantum circuit.
        fidelity = ComputeUncompute(sampler=sampler) # ComputeUncompute is a method that uses the compute-uncompute technique to estimate the fidelity.

        # The ZZFeatureMap maps classical data to a quantum state. The 'feature_dimension' parameter specifies the
        # dimensionality of the input data, and 'reps' indicates the number of repetitions of the feature map.
        feature_map = ZZFeatureMap(feature_dimension=self.n_qubits, reps=1)

        # Initialize the quantum kernel with the specified fidelity computation method and feature map.
        # The FidelityQuantumKernel calculates the kernel matrix for quantum data using the fidelity metric.
        qpca_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

        # Evaluate the kernel matrix for the training data.
        # x.detach().numpy() converts the input tensor 'x' to a numpy array.
        matrix_train = qpca_kernel.evaluate(x_vec=x.detach().numpy())

        # Initialize KernelPCA with the number of components and kernel type.
        # KernelPCA is a variant of Principal Component Analysis that uses kernel methods to perform
        # dimensionality reduction in high-dimensional feature spaces. The 'precomputed' kernel indicates
        # that we will provide a precomputed kernel matrix instead of calculating it internally.
        kernel_pca_q = KernelPCA(n_components=self.n_pca, kernel="precomputed")

        # Fit KernelPCA on the precomputed kernel matrix and transform the training data.
        # This step reduces the dimensionality of the training data to the specified number of components.
        train_features_q = kernel_pca_q.fit_transform(matrix_train)
        logits_pca = torch.tensor(train_features_q,dtype=torch.double)

        out = self.decoder(logits_pca).to(torch.float64) # The data are decoded back to 64 dimensions


        return out




    def forward_2(self, x):
        # Here we are stopping on the reduced weights before the KernelPCA. The output neurons of this method will be the input for the VQC
        x = self.input(x)




        return x