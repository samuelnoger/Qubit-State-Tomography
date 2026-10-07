# tomography/end_to_end_eval.py
"""End to end: states -> basis rotation -> noisy readout (confusion matrix of a classifier) -> counts -> reconstruction.

Compared on the same states and counts:
  naive MLE            : reported counts reconstructed as if readout were perfect
  corrected MLE        : MLE with effective operators E'[k,m] = sum_t C[m,t] E[k,t], using the exact C (idealized calibration)
  network (aware)      : network trained on counts generated through the same channel
  network (ideal)      : network trained on perfect-readout counts, applied naively (optional; expected to fail)

    python -m tomography.end_to_end_eval --ensemble near_bell --readout-json results/sweeps/readout_tomography.json \
        --classifier cnn --aware-checkpoint checkpoints/neural_tomo_aware/neural_tomo_near_bell_cnn.pth
"""
import argparse
import json
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from tomography.benchmark_speed import load_network, measurement_operators, fidelity, mle, ENSEMBLES, _states


def sample_counts_channel(rho, E, C, shots, rng):
    p = np.clip(_states.born_probabilities(rho, E), 0, None) @ C.T
    p = np.clip(p, 0, None)
    p /= p.sum(axis=1, keepdims=True)
    return np.stack([rng.multinomial(shots, p[k]) for k in range(p.shape[0])])


def infidelities(pred, true):
    return np.array([1 - fidelity(np.asarray(p, dtype=np.complex128), t) for p, t in zip(pred, true)])


def run(args, C, networks):
    rng = np.random.default_rng(args.seed + 4242)
    E = measurement_operators()
    E_eff = np.einsum('mt,ktij->kmij', C, E)                  # effective operators for reported outcomes
    results = {}
    for n_shots in args.shots:
        states = [ENSEMBLES[args.ensemble](rng) for _ in range(args.n_states)]
        counts = [sample_counts_channel(s, E, C, n_shots, rng) for s in states]
        x = np.array([(c / n_shots).reshape(-1) for c in counts], dtype=np.float32)
        true = np.array(states)

        res = {f"naive MLE ({args.mle_iters[-1]} it.)": infidelities([mle(c, E, iters=args.mle_iters[-1]) for c in counts], true)}
        for it in args.mle_iters:
            res[f"corrected MLE ({it} it.)"] = infidelities([mle(c, E_eff, iters=it) for c in counts], true)
        for name, predict in networks.items():
            res[name] = infidelities(predict(x, n_shots), true)
        results[n_shots] = res

        print(f"\n{args.ensemble} states, {n_shots} shots/setting, {args.n_states} states, channel: {args.classifier}")
        best_mle = min((m for m in res if m.startswith("corrected")), key=lambda m: res[m].mean())
        for method, inf in res.items():
            line = f"  {method:<34s}: {inf.mean():.3e} +- {inf.std() / np.sqrt(len(inf)):.1e}"
            if method.startswith("network"):
                d = inf - res[best_mle]
                line += f"   | vs best corrected MLE: x{inf.mean() / res[best_mle].mean():.2f} (paired {d.mean():+.1e} +- {d.std() / np.sqrt(len(d)):.1e})"
            print(line)
    return results


def plot(results, args):
    shots = list(results)
    methods = list(results[shots[0]])
    
    plt.figure(figsize=(9, 6))
    
    for m in methods:
        mean = [results[n][m].mean() for n in shots]
        sem = [results[n][m].std() / np.sqrt(len(results[n][m])) for n in shots]
        
        # Map method names to the requested colors and line styles
        if m.startswith("naive MLE"):
            fmt, color = 'o--', 'gray'
        elif m.startswith("corrected MLE"):
            fmt, color = 'o--', 'black'
        elif m.startswith("network (trained without"):
            fmt, color = 's-', 'lightcoral'
        elif m.startswith("network (aware"):
            fmt, color = 's-', 'darkred'
        else:
            fmt, color = 'v:', 'blue'
            
        plt.errorbar(shots, mean, yerr=sem, fmt=fmt, color=color, capsize=2, label=m)
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Shots per setting (N)')
    plt.ylabel('Mean Infidelity (1 - F)')
    plt.title(f"End to end: {args.ensemble} states through the '{args.classifier}' readout channel")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    path = os.path.join(args.results_dir, "figures", f"end_to_end_{args.ensemble}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300)
    print(f"\nSaved plot to {path}")


def main():
    p = argparse.ArgumentParser(description="End-to-end evaluation through the readout channel")
    p.add_argument("--ensemble", choices=list(ENSEMBLES), default="near_bell")
    p.add_argument("--readout-json", default="results/sweeps/readout_tomography.json")
    p.add_argument("--classifier", default="cnn")
    p.add_argument("--aware-checkpoint", default="checkpoints/neural_tomo_aware/neural_tomo_near_bell_cnn.pth", help="network trained on counts through the channel")
    p.add_argument("--ideal-checkpoint", default="checkpoints/neural_tomo_variable/neural_tomo_near_bell.pth", help="network trained on perfect-readout counts (optional)")
    p.add_argument("--shots", type=int, nargs="+", default=[10, 100, 1000, 10000])
    p.add_argument("--n-states", type=int, default=300)
    p.add_argument("--mle-iters", type=int, nargs="+", default=[50, 1000])
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", default="results/")
    args = p.parse_args()

    with open(args.readout_json) as f:
        C = np.array(json.load(f)["C_true"][args.classifier])
    networks = {}
    for name, path in (("network (aware of readout errors)", args.aware_checkpoint),
                       ("network (trained without readout errors)", args.ideal_checkpoint)):
        if path is not None:
            networks[name] = load_network(SimpleNamespace(checkpoint=path, hidden_dim=args.hidden_dim, n_layers=args.n_layers))

    results = run(args, C, networks)
    os.makedirs(os.path.join(args.results_dir, "sweeps"), exist_ok=True)
    with open(os.path.join(args.results_dir, "sweeps", f"end_to_end_{args.ensemble}.json"), "w") as f:
        json.dump({"ensemble": args.ensemble, "classifier": args.classifier, "n_states": args.n_states,
                   "mean": {str(n): {m: float(v.mean()) for m, v in r.items()} for n, r in results.items()},
                   "sem": {str(n): {m: float(v.std() / np.sqrt(len(v))) for m, v in r.items()} for n, r in results.items()}}, f, indent=2)
    plot(results, args)


if __name__ == "__main__":
    main()