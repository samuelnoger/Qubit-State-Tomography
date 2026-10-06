# utils/losses.py
import torch

def nll_loss(rho_pred, freq_true, E_tensor):
    """
    Negative Log-Likelihood loss.
    rho_pred: (B, 4, 4) complex tensor
    freq_true: (B, 36) true measurement frequencies
    E_tensor: (36, 4, 4) POVM measurement operators
    """
    # Calculate predicted probabilities: Tr(E_k * rho_pred)
    # E_tensor is (36, 4, 4), rho_pred is (B, 4, 4)
    # pred_probs shape: (B, 36)
    pred_probs = torch.einsum('kij,bji->bk', E_tensor, rho_pred).real
    
    # Clamp to prevent log(0)
    pred_probs = torch.clamp(pred_probs, min=1e-8, max=1.0)
    
    # Cross-entropy / NLL weighted by the true frequencies
    loss = -torch.sum(freq_true * torch.log(pred_probs), dim=-1)
    
    return torch.mean(loss)