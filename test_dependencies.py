import torch
import torchvision
import sklearn
from sklearn import __version__ as sklearn_version

print(f"PyTorch: {torch.__version__} ({'CPU' if not torch.cuda.is_available() else 'CUDA'})")
print(f"Torchvision: {torchvision.__version__}")
print(f"Scikit-learn: {sklearn_version}")

# Quick performance test
x = torch.randn(1000, 1000)
y = torch.mm(x, x)
print("CPU tensor multiplication works!")
print(f"Matrix shape: {y.shape}") 
