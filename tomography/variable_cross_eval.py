# tomography/variable_cross_eval.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.tomography_model import NeuralTomography
from utils.states import fidelity, measurement_operators, sample_counts
from utils.ensembles import random_broad_state, random_near_bell_state
from tomography.reconstruct import mle

def load_network(path, device):
    ckpt = torch.load(path, map_location=device)
    model = NeuralTomography(hidden_dim=ckpt["hidden_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    E = measurement_operators()
    rng = np.random.default_rng(42)
    
    shot_counts = [10, 31, 100, 316, 1000, 3162, 10000]
    n_states = 250  # Lowered slightly to speed up the MLE sweep
    
    # Path depends on your exact N_TRAIN and EPOCHS shell variables
    ckpt_dir = "checkpoints/neural_tomo_variable_200000_100ep"
    broad_model = load_network(f"{ckpt_dir}/neural_tomo_broad.pth", device)
    bell_model = load_network(f"{ckpt_dir}/neural_tomo_near_bell.pth", device)
    
    results = {"MLE (broad)": [], "NN (broad)": [], "MLE (near_bell)": [], "NN (near_bell)": []}
               
    for shots in shot_counts:
        print(f"\nEvaluating at N = {shots} shots...")
        log_N = np.log10(shots)
        
        # 1. Broad ensemble evaluation
        broad_states = [random_broad_state(rng) for _ in range(n_states)]
        mle_err, nn_err = [], []
        for rho in tqdm(broad_states, desc="Broad states"):
            counts = sample_counts(rho, E, shots, rng)
            freq = counts / shots
            
            mle_err.append(1 - fidelity(mle(counts, E, iters=1000), rho))
            
            x = np.zeros(37, dtype=np.float32)
            x[:36] = freq.flatten()
            x[36] = log_N
            with torch.no_grad():
                rho_nn = broad_model(torch.from_numpy(x).unsqueeze(0).to(device)).cpu().numpy()[0]
            nn_err.append(1 - fidelity(rho_nn, rho))
            
        results["MLE (broad)"].append(np.mean(mle_err))
        results["NN (broad)"].append(np.mean(nn_err))
        
        # 2. Near-Bell ensemble evaluation
        bell_states = [random_near_bell_state(rng) for _ in range(n_states)]
        mle_err, nn_err = [], []
        for rho in tqdm(bell_states, desc="Near-Bell states"):
            counts = sample_counts(rho, E, shots, rng)
            freq = counts / shots
            
            mle_err.append(1 - fidelity(mle(counts, E, iters=1000), rho))
            
            x = np.zeros(37, dtype=np.float32)
            x[:36] = freq.flatten()
            x[36] = log_N
            with torch.no_grad():
                rho_nn = bell_model(torch.from_numpy(x).unsqueeze(0).to(device)).cpu().numpy()[0]
            nn_err.append(1 - fidelity(rho_nn, rho))
            
        results["MLE (near_bell)"].append(np.mean(mle_err))
        results["NN (near_bell)"].append(np.mean(nn_err))
        
    plt.figure(figsize=(9, 6))
    plt.plot(shot_counts, results["MLE (broad)"], 'o--', color='gray', label='MLE (broad states)')
    plt.plot(shot_counts, results["NN (broad)"], 's-', color='black', label='NN (broad states)')
    
    plt.plot(shot_counts, results["MLE (near_bell)"], 'o--', color='lightcoral', label='MLE (near_bell states)')
    plt.plot(shot_counts, results["NN (near_bell)"], 's-', color='darkred', label='NN (near_bell states)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Shots per setting (N)')
    plt.ylabel('Mean Infidelity (1 - F)')
    plt.title('Variable-Shot Network vs MLE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/figures/variable_shot_eval.png', dpi=300)
    print("\nSaved plot to variable_shot_eval.png")

if __name__ == "__main__":
    main()