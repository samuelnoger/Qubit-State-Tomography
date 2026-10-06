# data/generate_neural_tomo_data.py
import os
import torch
import numpy as np

from train.arguments import parse_args
from utils.states import (measurement_operators, random_pure_state, 
                               random_mixed_state, bell_state, sample_counts)

def generate_dataset(n_samples, state_type, shots, rng, E, rank=4, noise=0.0):
    x = np.zeros((n_samples, 36), dtype=np.float32)
    y = np.zeros((n_samples, 4, 4), dtype=np.complex128)
    
    for i in range(n_samples):
        if state_type == "pure":
            rho = random_pure_state(rng)
        elif state_type == "mixed":
            rho = random_mixed_state(rng, rank=rank)
        elif state_type == "bell":
            rho_bell = bell_state(["phi+", "phi-", "psi+", "psi-"][rng.integers(4)])
            rho = (1 - noise) * rho_bell + noise * np.eye(4) / 4
        else:
            raise ValueError(f"Unknown state type {state_type}")
            
        counts = sample_counts(rho, E, shots, rng)
        freq = counts / shots
        
        x[i] = freq.flatten()
        y[i] = rho
        
    return torch.from_numpy(x), torch.from_numpy(y)

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    E = measurement_operators()

    print(f"Generating {args.n_train_tomo + args.n_val_tomo + args.n_test_tomo} {args.tomo_state_type} states at {args.tomo_shots} shots/setting...")
    
    x_tr, y_tr = generate_dataset(args.n_train_tomo, args.tomo_state_type, args.tomo_shots, rng, E, args.mixed_rank, args.bell_noise)
    x_va, y_va = generate_dataset(args.n_val_tomo, args.tomo_state_type, args.tomo_shots, rng, E, args.mixed_rank, args.bell_noise)
    x_te, y_te = generate_dataset(args.n_test_tomo, args.tomo_state_type, args.tomo_shots, rng, E, args.mixed_rank, args.bell_noise)

    os.makedirs(os.path.dirname(args.tomo_data_path) or ".", exist_ok=True)
    
    torch.save({
        "train": {"x": x_tr, "y": y_tr},
        "val": {"x": x_va, "y": y_va},
        "test": {"x": x_te, "y": y_te},
        "info": {"shots": args.tomo_shots, "state_type": args.tomo_state_type}
    }, args.tomo_data_path)
    print(f"Saved {args.tomo_data_path}")

if __name__ == "__main__":
    main()