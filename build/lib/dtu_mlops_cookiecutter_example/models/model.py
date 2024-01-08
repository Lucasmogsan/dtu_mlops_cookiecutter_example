import torch
from torch import nn, optim
import torch.nn.functional as F


myawesomemodel = nn.Sequential(
    nn.Conv2d(1, 32, 3), # [B, 1, 28, 28] -> [B, 32, 26, 26]
    nn.LeakyReLU(),
    nn.MaxPool2d(2), # [B, 32, 26, 26] -> [B, 32, 13, 13]
    nn.Conv2d(32, 64, 3), # [B, 32, 13, 13] -> [B, 64, 11, 11]
    nn.ReLU(),
    nn.MaxPool2d(2), # [B, 64, 11, 11] -> [B, 64, 5, 5]
    nn.Flatten(), # [B, 64, 5, 5] -> [B, 1600]
    nn.Linear(1600, 10), # [B, 1600] -> [B, 10]
    nn.Softmax(dim=1) # Apply softmax activation function
    )


if __name__ == "__main__":
    print("Test of My awesome model")

    # Test of model (with random input)
    test_input = torch.randn(1, 1, 28, 28)
    test_output = myawesomemodel(test_input)
    print(test_output.shape)
