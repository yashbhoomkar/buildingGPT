import torch

print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())


device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)