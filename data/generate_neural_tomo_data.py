# data/generate_neural_tomo_data.py
import os
import torch
import numpy as np

from train.arguments import parse_args
from utils.states import measurement_operators, sample_counts
from utils.ensembles import random_broad_state, random_near_bell_state

def generate_dataset(n_samples, state_type, min_log_shots, max_log_shots, rng, E):
    # 37 features: 36 frequencies + 1 log10(N)
    x = np.zeros((n_samples, 37), dtype=np.float32)
    y = np.zeros((n_samples, 4, 4), dtype=np.complex128)
    
    if state_type == "broad":
        gen_state = lambda rng: random_broad_state(rng, ranks=(1, 2, 3, 4))
    elif state_type == "near_bell":
        gen_state = lambda rng: random_near_bell_state(rng)
    else:
        raise ValueError(f"Unknown state type: {state_type}")
        
    for i in range(n_samples):
        rho = gen_state(rng)
        
        # Sample N logarithmically
        log_N = rng.uniform(min_log_shots, max_log_shots)
        shots = int(np.round(10**log_N))
        
        counts = sample_counts(rho, E, shots, rng)
        freq = counts / shots
        
        x[i, :36] = freq.flatten()
        x[i, 36] = log_N
        y[i] = rho
        
    return torch.from_numpy(x), torch.from_numpy(y)

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    E = measurement_operators()

    min_log_shots, max_log_shots = 1.0, 4.0

    print(f"Generating {args.n_train_tomo + args.n_val_tomo + args.n_test_tomo} {args.tomo_state_type} states across variable shots...")
    
    x_tr, y_tr = generate_dataset(args.n_train_tomo, args.tomo_state_type, min_log_shots, max_log_shots, rng, E)
    x_va, y_va = generate_dataset(args.n_val_tomo, args.tomo_state_type, min_log_shots, max_log_shots, rng, E)
    x_te, y_te = generate_dataset(args.n_test_tomo, args.tomo_state_type, min_log_shots, max_log_shots, rng, E)

    os.makedirs(os.path.dirname(args.tomo_data_path) or ".", exist_ok=True)
    
    torch.save({
        "train": {"x": x_tr, "y": y_tr},
        "val": {"x": x_va, "y": y_va},
        "test": {"x": x_te, "y": y_te},
        "info": {"state_type": args.tomo_state_type}
    }, args.tomo_data_path)
    print(f"Saved {args.tomo_data_path}")

if __name__ == "__main__":
    main()