
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dims=[128, 64]):
        super(BCModel, self).__init__()
        layers = []
        last_dim = input_size
        for dim in hidden_dims:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.ReLU())
            last_dim = dim
        layers.append(nn.Linear(last_dim, num_classes))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = BCModel(5, 100)
    print(model)
    dummy_input = torch.randn(1, 5)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
