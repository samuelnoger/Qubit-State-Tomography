# model/tomography_model.py
import torch
import torch.nn as nn
from utils.features import feature_matrix

class NeuralTomography(nn.Module):
    def __init__(self, hidden_dim=256, n_layers=3):
        super().__init__()
        self.register_buffer("M", torch.tensor(feature_matrix(), dtype=torch.float32))
        self.register_buffer("eye", torch.eye(4))
        
        # MLP input dimension is 16 (15 correlators + 1 log10(N))
        layers, d = [], 16
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
            d = hidden_dim
            
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 32)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x is (B, 37). Split into frequencies and log_N
        freqs = x[:, :36]
        log_N = (x[:, 36:] - 2.5) / 1.5
        
        # Project frequencies and concatenate log_N
        correlators = freqs @ self.M
        mlp_in = torch.cat([correlators, log_N], dim=1)
        
        out = self.head(self.body(mlp_in))
        
        gr = self.eye + out[:, :16].reshape(-1, 4, 4)
        gi = out[:, 16:].reshape(-1, 4, 4)
        re = gr @ gr.transpose(1, 2) + gi @ gi.transpose(1, 2)
        im = gi @ gr.transpose(1, 2) - gr @ gi.transpose(1, 2)
        tr = torch.diagonal(re, dim1=-2, dim2=-1).sum(-1).reshape(-1, 1, 1)
        
        return torch.complex(re / tr, im / tr)