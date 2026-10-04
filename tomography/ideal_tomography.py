# tomography/ideal_tomography.py
"""Milestone 1: ideal two-qubit tomography. Infidelity vs. number of shots for linear inversion and MLE.

    PYTHONPATH=. python -m tomography.ideal_tomography
    PYTHONPATH=. python -m tomography.ideal_tomography --n-states 20 --shots 30 300 3000
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from tomography.states import (measurement_operators, random_pure_state, random_mixed_state,
                               bell_state, sample_counts, fidelity)
from tomography.reconstruct import linear_inversion, project_to_physical, mle
from train.arguments import parse_args

BELL_NAMES = ["phi+", "phi-", "psi+", "psi-"]
TITLES = {"pure": "Random pure states", "mixed": "Random mixed states", "bell": "Bell states"}


def make_state(kind, rng, rank):
    if kind == "pure":
        return random_pure_state(rng)
    if kind == "mixed":
        return random_mixed_state(rng, rank)
    return bell_state(BELL_NAMES[rng.integers(4)])


def run(args):
    rng = np.random.default_rng(args.seed)
    E = measurement_operators()
    results = {}
    for kind in args.state_types:
        results[kind] = {"li": [], "mle": []}                     # per shot number -> list over states
        for n in args.shots:
            li, ml = [], []
            for _ in range(args.n_states):
                rho = make_state(kind, rng, args.mixed_rank)
                counts = sample_counts(rho, E, n, rng)
                li.append(1 - fidelity(project_to_physical(linear_inversion(counts)), rho))
                ml.append(1 - fidelity(mle(counts, E, iters=args.mle_iters), rho))
            results[kind]["li"].append(li)
            results[kind]["mle"].append(ml)
            print(f"{kind:>5s}  {n:>6d} shots/setting:  1-F  linear = {np.mean(li):.2e}   MLE = {np.mean(ml):.2e}")
    return results


def plot(results, args):
    kinds = list(results)
    fig, axes = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 4), sharey=True, squeeze=False)
    shots = np.array(args.shots)
    for ax, kind in zip(axes[0], kinds):
        for key, label, marker in (("li", "Linear inversion + projection", "o"), ("mle", "Maximum likelihood", "s")):
            arr = np.array(results[kind][key])                      # (n_shots, n_states)
            mean, sem = arr.mean(1), arr.std(1) / np.sqrt(arr.shape[1])
            ax.errorbar(shots, mean, yerr=sem, marker=marker, capsize=3, label=label)
        ref = np.mean(results[kind]["mle"][0]) * shots[0] / shots   # 1/N guide through the first MLE point
        ax.plot(shots, ref, "k--", linewidth=1, label=r"$\propto 1/N$")
        ref2 = np.mean(results[kind]["li"][0]) * np.sqrt(shots[0] / shots)    # 1/sqrt(N) guide through the first linear-inversion point
        ax.plot(shots, ref2, "k:", linewidth=1, label=r"$\propto 1/\sqrt{N}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(TITLES[kind]); ax.set_xlabel("Shots per measurement setting")
        
    axes[0][0].set_ylabel("Infidelity 1 - F")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Ideal two-qubit tomography (9 settings, {args.n_states} states per point)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.results_dir, "figures", "ideal_tomography.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")


def main():
    args = parse_args()
    results = run(args)
    os.makedirs(os.path.join(args.results_dir, "sweeps"), exist_ok=True)
    with open(os.path.join(args.results_dir, "sweeps", "ideal_tomography.json"), "w") as f:
        json.dump({"shots": args.shots, "n_states": args.n_states, "results": results}, f)
    plot(results, args)


if __name__ == "__main__":
    main()