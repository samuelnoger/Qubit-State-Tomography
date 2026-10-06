# tomography/neural_cross_eval.py
"""Cross-evaluation of the neural reconstruction: how much does the learned prior cost off-distribution?

Fresh random states of rank 1 (pure), 2, 3 and 4 (full-rank mixed) are measured with the same number of shots per
setting as the networks were trained on. The same states and counts go to linear inversion, MLE, and each network.

    python -m tomography.neural_cross_eval --ckpt-dir checkpoints/neural_tomo_1000/ --shots 1000 --n-states 1000
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.states import measurement_operators, random_mixed_state, sample_counts, fidelity
from tomography.reconstruct import linear_inversion, project_to_physical, mle

RANK_NAMES = {1: "pure (rank 1)", 2: "rank 2", 3: "rank 3", 4: "mixed (rank 4)"}


def load_networks(ckpt_dir, names, shots):
    """Load neural_tomo_<name>.pth for each name. Returns {name: predict(x: (n, 36) float32) -> (n, 4, 4) complex}."""
    import torch
    from model.tomography_model import NeuralTomography

    networks = {}
    for name in names:
        path = os.path.join(ckpt_dir, f"neural_tomo_{name}.pth")
        ck = torch.load(path, map_location="cpu")
        trained_shots = ck.get("info", {}).get("shots")
        if trained_shots is not None and trained_shots != shots:
            raise ValueError(f"{path} was trained at {trained_shots} shots/setting, but --shots is {shots}")
        model = NeuralTomography(hidden_dim=ck.get("hidden_dim", 256))
        model.load_state_dict(ck["state_dict"])
        model.eval()

        def predict(x, model=model):
            out = []
            with torch.no_grad():
                for i in range(0, len(x), 2048):
                    out.append(model(torch.from_numpy(x[i:i + 2048])).numpy())
            return np.concatenate(out)

        networks[name] = predict
    return networks


def infidelities(pred, true):
    """Per-state infidelity, in double precision."""
    return np.array([1 - fidelity(np.asarray(p, dtype=np.complex128), t) for p, t in zip(pred, true)])


def run(args, networks):
    rng = np.random.default_rng(args.seed + 777)               # different from the training data seeds
    E = measurement_operators()
    results = {}
    for rank in args.ranks:
        states = [random_mixed_state(rng, rank=rank) for _ in range(args.n_states)]
        counts = [sample_counts(s, E, args.shots, rng) for s in states]
        x = np.array([(c / args.shots).reshape(-1) for c in counts], dtype=np.float32)
        true = np.array(states)

        res = {"linear inversion": infidelities([project_to_physical(linear_inversion(c)) for c in counts], true),
               "MLE": infidelities([mle(c, E, iters=args.mle_iters) for c in counts], true)}
        for name, predict in networks.items():
            res[f"network trained on {name}"] = infidelities(predict(x), true)
        results[rank] = res

        print(f"\nTest states: {RANK_NAMES[rank]}, {args.n_states} states, {args.shots} shots/setting")
        for method, inf in res.items():
            line = f"  {method:<28s}: {inf.mean():.3e} +- {inf.std() / np.sqrt(len(inf)):.1e}"
            if method.startswith("network"):
                d = inf - res["MLE"]                           # paired: same states, same counts
                line += f"   | vs MLE: x{inf.mean() / res['MLE'].mean():.2f} (paired difference {d.mean():+.1e} +- {d.std() / np.sqrt(len(d)):.1e})"
            print(line)
    return results


def plot(results, args):
    ranks = list(results)
    methods = list(results[ranks[0]])
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, method in enumerate(methods):
        means = [results[r][method].mean() for r in ranks]
        sems = [results[r][method].std() / np.sqrt(len(results[r][method])) for r in ranks]
        
        # Solid lines for networks, dashed for baselines (MLE, Linear Inversion)
        linestyle = '-' if 'network' in method else '--'
        
        ax.errorbar(
            ranks, 
            means, 
            yerr=sems, 
            label=method,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            capsize=4,
            markersize=6,
            linewidth=1.5,
            linestyle=linestyle
        )
        
    ax.set_xticks(ranks)
    ax.set_xticklabels([RANK_NAMES[r] for r in ranks])
    ax.set_yscale("log")
    ax.set_xlabel("Test states")
    ax.set_ylabel("Mean Infidelity (1 - F)")
    ax.set_title(f"Neural reconstruction on states it was and was not trained on ({args.shots} shots/setting)", fontsize=10)
    
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    path = os.path.join(args.results_dir, "figures", "neural_cross_eval.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    print(f"\nSaved {path}")


def main():
    p = argparse.ArgumentParser(description="Cross-evaluation of the neural tomography networks")
    p.add_argument("--ckpt-dir", required=True, help="folder with neural_tomo_pure.pth and neural_tomo_mixed.pth")
    p.add_argument("--networks", nargs="+", default=["pure", "mixed"])
    p.add_argument("--shots", type=int, default=1000, help="shots per setting; must match the training")
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--n-states", type=int, default=1000)
    p.add_argument("--mle-iters", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", default="results/")
    args = p.parse_args()

    networks = load_networks(args.ckpt_dir, args.networks, args.shots)
    results = run(args, networks)
    os.makedirs(os.path.join(args.results_dir, "sweeps"), exist_ok=True)
    with open(os.path.join(args.results_dir, "sweeps", "neural_cross_eval.json"), "w") as f:
        json.dump({"shots": args.shots, "n_states": args.n_states,
                   "mean": {str(r): {m: float(v.mean()) for m, v in res.items()} for r, res in results.items()},
                   "sem": {str(r): {m: float(v.std() / np.sqrt(len(v))) for m, v in res.items()} for r, res in results.items()}}, f, indent=2)
    plot(results, args)


if __name__ == "__main__":
    main()