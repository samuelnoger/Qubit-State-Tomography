# train/train_neural_tomo.py
import json
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from model.tomography_model import NeuralTomography
from utils.states import fidelity, measurement_operators
from tomography.reconstruct import linear_inversion, project_to_physical, mle
from train.arguments import parse_args

def complex_mse_loss(rho_pred, rho_true):
    diff = rho_pred - rho_true
    return torch.sum(diff.real ** 2 + diff.imag ** 2) / rho_pred.shape[0]

@torch.no_grad()
def predict(model, x, device, bs=2048):
    model.eval()
    return torch.cat([model(x[i:i + bs].to(device)).cpu() for i in range(0, len(x), bs)]).numpy()

def mean_infidelity(pred, true):
    return float(np.mean([1 - fidelity(p.astype(np.complex128), t.astype(np.complex128)) for p, t in zip(pred, true)]))

def baselines(x, y, n=500, mle_iters=1000):
    E = measurement_operators()
    x, y = x[:n].numpy(), y[:n].numpy()
    out = {"I/4 (ignores the data)": mean_infidelity(np.broadcast_to(np.eye(4) / 4, y.shape), y)}
    li, ml = [], []
    for inputs, rho in zip(x, y):
        freq = inputs[:36]
        log_N = inputs[36]
        shots = int(np.round(10**log_N))
        
        counts = np.rint(freq.reshape(9, 4) * shots)
        li.append(project_to_physical(linear_inversion(counts)))
        ml.append(mle(counts, E, iters=mle_iters))
        
    out["linear inversion + projection"] = mean_infidelity(np.array(li), y)
    out["maximum likelihood"] = mean_infidelity(np.array(ml), y)
    return out

def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")

    data = torch.load(args.tomo_data_path)
    tr, va, te = data["train"], data["val"], data["test"]
    print(f"Training unified neural tomography on {args.tomo_data_path} (Variable Shots)")

    print("\nReference infidelities on the first 500 validation states:")
    for name, val in baselines(va["x"], va["y"]).items():
        print(f"  {name:<32s}: {val:.4e}")

    loader = DataLoader(TensorDataset(tr["x"], tr["y"]), batch_size=args.batch_size, shuffle=True)
    model = NeuralTomography(hidden_dim=args.tomo_hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    best_inf, best_state, history = float("inf"), None, []
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(torch.complex64).to(device)
            optimizer.zero_grad()
            loss = complex_mse_loss(model(xb), yb)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * xb.size(0)
        scheduler.step()

        val_pred, val_true = predict(model, va["x"][:2000], device), va["y"][:2000].numpy()
        val_inf = mean_infidelity(val_pred, val_true)
        val_loss = float(np.mean(np.sum(np.abs(val_pred - val_true) ** 2, axis=(1, 2))))
        history.append({"epoch": epoch + 1, "train_loss": running / len(tr["x"]), "val_loss": val_loss, "val_infidelity": val_inf})
        
        if val_inf < best_inf:
            best_inf = val_inf
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        pbar.set_postfix({"Loss": f"{running / len(tr['x']):.3e}", "ValLoss": f"{val_loss:.3e}", "ValInfid": f"{val_inf:.3e}"})

    model.load_state_dict(best_state)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    state_type = data["info"]["state_type"]
    save_path = os.path.join(args.checkpoint_dir, f"neural_tomo_{state_type}.pth")
    torch.save({"state_dict": best_state, "hidden_dim": args.tomo_hidden_dim, "info": data["info"]}, save_path)

    # Save the training history for capacity/overfitting analysis
    history_path = os.path.join(args.checkpoint_dir, f"history_{state_type}.json")
    with open(history_path, "w") as f:
        json.dump(history, f)

    print("\nTest set, first 500 states:")
    for name, val in baselines(te["x"], te["y"]).items():
        print(f"  {name:<32s}: {val:.4e}")
    n = 500
    print(f"  {'neural network':<32s}: {mean_infidelity(predict(model, te['x'][:n], device), te['y'][:n].numpy()):.4e}")
    print(f"\nSaved to {save_path}")

if __name__ == "__main__":
    main()