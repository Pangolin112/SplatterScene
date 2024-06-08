import torch

# Example matrix
A = torch.randn(5, 3, requires_grad=True)

# Perform SVD
U, S, V = torch.svd(A)

# Define a simple function of the singular values
loss = torch.sum(S) + torch.sum(U) + torch.sum(V)

# Backpropagate
loss.backward()

# Print gradients
print(A.grad)
