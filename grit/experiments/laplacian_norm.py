import torch
import numpy as np


n = 5
A = torch.ones((n,n))

D = torch.diag(A.sum(dim=1))
inv_D = 1. / D
inv_D[inv_D == torch.inf] = 0.
sq_inv_D = inv_D.sqrt()

L = torch.eye(n) - sq_inv_D @ A @ sq_inv_D

print("Norm of laplacian matrix is", torch.linalg.norm(L))
print("The eigenvectors / values are, ", np.linalg.eigh(L))