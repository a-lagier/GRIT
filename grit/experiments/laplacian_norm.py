import torch
import numpy as np


n = 5
p = .8
A = (torch.rand(n,n) > p).float()

D = torch.diag(A.sum(dim=1))
inv_D = 1. / D
inv_D[inv_D == torch.inf] = 0.
sq_inv_D = inv_D.sqrt()

L = torch.eye(n) - sq_inv_D @ A @ sq_inv_D

lambdas = np.linalg.eigh(L).eigenvalues

P = 20
pow_norm_list = [torch.from_numpy(lambdas).pow(2*p_).sum().sqrt() / 2**p_ for p_ in range(P)]

print("Norm of laplacian matrix is", torch.linalg.norm(L))
print("The eigenvectors / values are, ", np.linalg.eigh(L))
print("Computed norm is", pow_norm_list)