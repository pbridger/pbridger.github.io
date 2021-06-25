import torch
from torchvision.transforms.functional import gaussian_blur
from nvtx_range import nvtx_range

class Net(torch.nn.Module):
    def forward(self, X):
        with nvtx_range('Net.forward'):
            with nvtx_range('normalize'):
                X = X.half().divide(255)
            with nvtx_range('blur'):
                return gaussian_blur(X, kernel_size=5, sigma=3)
