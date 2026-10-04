# model/readout_model.py
import torch
import torch.nn as nn


class ReadoutCNN(nn.Module):
    """1D CNN over the multi-channel record. Input (B, T, C) -> logits (B, n_outputs).

    C = 2 * n_qubits (I and Q per resonator), one logit per qubit.
    No global pooling on purpose: *when* the signal switches (a mid-readout decay)
    is the information that beats a matched filter, so we keep time position by flattening.
    """
    def __init__(self, seq_len, in_channels=2, n_outputs=1, channels=(16, 32, 32), hidden=64):
        super().__init__()
        c1, c2, c3 = channels
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, c1, 5, padding=2), nn.ReLU(),
            nn.Conv1d(c1, c2, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(c2, c3, 5, stride=2, padding=2), nn.ReLU(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, in_channels, seq_len)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, hidden), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(hidden, n_outputs))

    def forward(self, x):                       # x: (B, T, C)
        return self.head(self.conv(x.transpose(1, 2)))


class ReadoutGRU(nn.Module):
    """Small GRU alternative. Input (B, T, C) -> logits (B, n_outputs)."""
    def __init__(self, in_channels=2, n_outputs=1, hidden=64, layers=1):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n_outputs)

    def forward(self, x):
        _, h = self.gru(x)                      # h: (layers, B, hidden)
        return self.head(h[-1])