import torch.nn as nn
import torch

m = nn.MaxPool1d(3, stride=2)
input = torch.randn(5, 4, 5)
output = m(input)

print(input.size())
print(output.size())