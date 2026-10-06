# model/tomography_model.py
import torch
import torch.nn as nn

from utils.features import feature_matrix


class NeuralTomography(nn.Module):
    """MLP mapping 36 measured frequencies to a 4x4 density matrix.

    Differences from a plain lower-triangular Cholesky head:
      * Inputs are the 15 Pauli correlators (a fixed linear map of the frequencies), centred around 0.
      * rho = G G^dagger / Tr(G G^dagger) with a FULL complex 4x4 matrix G = I + output. A triangular
        Cholesky factor has entries ~ rho_ij / sqrt(rho_00), which blow up for states with small rho_00 and are
        hard to regress; G = sqrt(rho) is a smooth function of rho, also for pure states.
      * The head starts near zero, so training starts from G = I, i.e. rho = I/4, with well-scaled gradients.
      * No epsilon in the normalisation: Tr(G G^dagger) cannot collapse towards 0 while G stays close to I.
    All arithmetic is real; the complex tensor is only assembled at the end.
    """
    def __init__(self, hidden_dim=256, n_layers=3):
        super().__init__()
        self.register_buffer("M", torch.tensor(feature_matrix(), dtype=torch.float32))
        self.register_buffer("eye", torch.eye(4))
        layers, d = [], 15
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
            d = hidden_dim
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 32)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):                                        # x: (B, 36) frequencies
        out = self.head(self.body(x @ self.M))
        gr = self.eye + out[:, :16].reshape(-1, 4, 4)
        gi = out[:, 16:].reshape(-1, 4, 4)
        re = gr @ gr.transpose(1, 2) + gi @ gi.transpose(1, 2)    # Re(G G^dagger)
        im = gi @ gr.transpose(1, 2) - gr @ gi.transpose(1, 2)    # Im(G G^dagger)
        tr = torch.diagonal(re, dim1=-2, dim2=-1).sum(-1).reshape(-1, 1, 1)
        return torch.complex(re / tr, im / tr)