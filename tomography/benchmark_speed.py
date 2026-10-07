# tomography/benchmark_speed.py
"""Speed and convergence benchmark for channel-aware networks vs corrected classical estimators.

    python -m tomography.benchmark_speed --ensemble near_bell \
        --aware-checkpoint checkpoints/neural_tomo_aware/neural_tomo_near_bell_cnn.pth
"""
import argparse
import json
import time
import numpy as np

from utils.states import measurement_operators, fidelity, born_probabilities
from utils.ensembles import random_broad_state, random_near_bell_state
from tomography.reconstruct import mle, parametric_mle

ENSEMBLES = {"broad": random_broad_state, "near_bell": random_near_bell_state}


def sample_counts_channel(rho, E, C, shots, rng):
    p = np.clip(born_probabilities(rho, E), 0, None) @ C.T
    p = np.clip(p, 0, None)
    p /= p.sum(axis=1, keepdims=True)
    return np.stack([rng.multinomial(shots, p[k]) for k in range(p.shape[0])])


def load_network(checkpoint, hidden_dim=256, n_layers=3):
    if checkpoint is None:
        return None
    import torch
    from model.tomography_model import NeuralTomography

    ck = torch.load(checkpoint, map_location="cpu")
    wrapped = isinstance(ck, dict) and "state_dict" in ck
    hidden = ck.get("hidden_dim", hidden_dim) if wrapped else hidden_dim
    n_layers = ck.get("n_layers", n_layers) if wrapped else n_layers
    model = NeuralTomography(hidden_dim=hidden, n_layers=n_layers)
    model.load_state_dict(ck["state_dict"] if wrapped else ck)
    model.eval()

    def predict(x, n_shots):
        x37 = np.concatenate([x, np.full((len(x), 1), np.log10(n_shots), dtype=np.float32)], axis=1)
        with torch.no_grad():
            return model(torch.from_numpy(x37)).numpy()

    return predict


def mean_infidelity(pred, true):
    return float(np.mean([1 - fidelity(np.asarray(p, dtype=np.complex128), t) for p, t in zip(pred, true)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble", choices=list(ENSEMBLES), default="broad")
    p.add_argument("--aware-checkpoint", required=True, help="trained aware network (.pth)")
    p.add_argument("--readout-json", default="results/sweeps/readout_tomography.json")
    p.add_argument("--classifier", default="cnn")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--shots", type=int, nargs="+", default=[100, 1000, 10000])
    p.add_argument("--n-states", type=int, default=100)
    p.add_argument("--iters", type=int, nargs="+", default=[50, 1000])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    E = measurement_operators()

    with open(args.readout_json) as f:
        C = np.array(json.load(f)["C_true"][args.classifier])
    
    # Calculate effective POVM operators for corrected classical estimators
    E_eff = np.einsum('mt,ktij->kmij', C, E)

    predict = load_network(args.aware_checkpoint, args.hidden_dim, args.n_layers)

    for n_shots in args.shots:
        states = [ENSEMBLES[args.ensemble](rng) for _ in range(args.n_states)]
        counts = [sample_counts_channel(s, E, C, n_shots, rng) for s in states]
        print(f"\n{args.ensemble} states, {n_shots} shots/setting, {args.n_states} states, channel: {args.classifier}")
        
        t_mle = None
        for iters in args.iters:
            t0 = time.perf_counter()
            est = [mle(c, E_eff, iters=iters, tol=0) for c in counts]
            t_mle = (time.perf_counter() - t0) / len(counts)
            print(f"  corrected MLE, {iters:>4d} it. : {t_mle * 1e3:8.2f} ms/state   infidelity {mean_infidelity(est, states):.3e}")

        t_param = None
        if args.ensemble == "near_bell":
            t0 = time.perf_counter()
            est_param = [parametric_mle(c, E_eff) for c in counts]
            t_param = (time.perf_counter() - t0) / len(counts)
            print(f"  Parametric MLE           : {t_param * 1e3:8.2f} ms/state   infidelity {mean_infidelity(est_param, states):.3e}")

        if predict is not None:
            x = np.array([(c / n_shots).reshape(-1) for c in counts], dtype=np.float32)
            predict(x[:8], n_shots)  # warm-up
            
            reps = 5
            t0 = time.perf_counter()
            for _ in range(reps):
                out = predict(x, n_shots)
            t_batch = (time.perf_counter() - t0) / reps / len(x)
            
            n1 = min(50, len(x))
            t0 = time.perf_counter()
            for i in range(n1):
                predict(x[i:i + 1], n_shots)
            t_single = (time.perf_counter() - t0) / n1
            
            print(f"  network, batch of {len(x):<4d}   : {t_batch * 1e6:8.1f} us/state   infidelity {mean_infidelity(out, states):.3e}")
            print(f"  network, one state       : {t_single * 1e6:8.1f} us/state")
            print(f"  speedup vs. MLE ({args.iters[-1]} it) : x{t_mle / t_batch:,.0f} (batched), x{t_mle / t_single:,.0f} (single)")
            if t_param is not None:
                print(f"  speedup vs. Param. MLE   : x{t_param / t_batch:,.0f} (batched), x{t_param / t_single:,.0f} (single)")


if __name__ == "__main__":
    main()