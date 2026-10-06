# tomography/readout_tomography.py
"""Milestone 2: tomography through a noisy readout chain.

For each classifier, compare reconstruction with
  ideal     : perfect readout (the milestone 1 benchmark)
  naive     : counts from the classifier, reconstructed as if readout were perfect
  corrected : MLE with the effective measurement operators E'[k,m] = sum_t C_cal[m,t] E[k,t]

    python -m tomography.readout_tomography --shots 100 1000 10000 100000
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.states import (measurement_operators, random_pure_state, random_mixed_state,
                               bell_state, sample_counts, fidelity)
from tomography.reconstruct import mle
from utils.readout_chain import build_chain, effective_operators, sample_counts_readout
from train.arguments import parse_args

BELL_NAMES = ["phi+", "phi-", "psi+", "psi-"]
TITLES = {"pure": "Random pure states", "mixed": "Random mixed states", "bell": "Bell states"}
NICE = { "cnn": "1D CNN","integrated": "Integrated threshold", "matched": "Matched filter", "lda_indep": "LDA (own channels)",
        "lda": "Linear (LDA)"}


def make_state(kind, rng, rank):
    if kind == "pure":
        return random_pure_state(rng)
    if kind == "mixed":
        return random_mixed_state(rng, rank)
    return bell_state(BELL_NAMES[rng.integers(4)])


def run(args, chain):
    rng = np.random.default_rng(args.seed)
    E = measurement_operators()
    E_eff = {name: effective_operators(E, c["C_cal"]) for name, c in chain.items()}
    results = {}
    for kind in args.state_types:
        res = {"ideal": [], "naive": {n: [] for n in chain}, "corrected": {n: [] for n in chain}}
        for n_shots in args.shots:
            ideal, naive, corr = [], {n: [] for n in chain}, {n: [] for n in chain}
            for _ in range(args.n_states):
                rho = make_state(kind, rng, args.mixed_rank)
                ideal.append(1 - fidelity(mle(sample_counts(rho, E, n_shots, rng), E, iters=args.mle_iters), rho))
                for name, c in chain.items():
                    counts = sample_counts_readout(rho, E, c["C_true"], n_shots, rng)
                    naive[name].append(1 - fidelity(mle(counts, E, iters=args.mle_iters), rho))
                    corr[name].append(1 - fidelity(mle(counts, E_eff[name], iters=args.mle_iters), rho))
            res["ideal"].append(ideal)
            for name in chain:
                res["naive"][name].append(naive[name])
                res["corrected"][name].append(corr[name])
            line = "  ".join(f"{nm}: naive {np.mean(naive[nm]):.1e} / corr {np.mean(corr[nm]):.1e}" for nm in chain)
            print(f"{kind:>5s} {n_shots:>7d} shots:  ideal {np.mean(ideal):.1e} | {line}")
        results[kind] = res
    return results


def plot(results, chain, args):
    kinds = list(results)
    shots = np.array(args.shots)
    fig, axes = plt.subplots(2, len(kinds), figsize=(5 * len(kinds), 7), sharex=True, sharey=True, squeeze=False)
    for col, kind in enumerate(kinds):
        for row, method in enumerate(("naive", "corrected")):
            ax = axes[row][col]
            arr = np.array(results[kind]["ideal"])
            ax.errorbar(shots, arr.mean(1), yerr=arr.std(1) / np.sqrt(arr.shape[1]), color="k", marker="o",
                        capsize=2, markersize=3, linewidth=1, label="ideal readout")
            for name in chain:
                arr = np.array(results[kind][method][name])
                label = f"{NICE.get(name, name)} (F={chain[name]['fid'].mean():.3f})"
                ax.errorbar(shots, arr.mean(1), yerr=arr.std(1) / np.sqrt(arr.shape[1]), marker="s", capsize=2, label=label, markersize=3, linewidth=1)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(shots.min() / 3, shots.max() * 3)
            if row == 0:
                ax.set_title(TITLES[kind])
            if row == 1:
                ax.set_xlabel("Shots per measurement setting")
            if col == 0:
                ax.set_ylabel("Infidelity 1 - F\n" + ("(readout errors ignored)" if method == "naive" else "(corrected with calibrated C)"))
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"Two-qubit tomography through the readout chain ({args.n_states} states per point, {args.n_cal} calibration records per state)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.results_dir, "figures", "readout_tomography.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")


def main():
    args = parse_args()
    json_path = os.path.join(args.results_dir, "sweeps", "readout_tomography.json")
    if args.replot:                                    # redraw from saved results, no simulation
        with open(json_path) as f:
            d = json.load(f)
        args.shots, args.n_states, args.n_cal = d["shots"], d["n_states"], d["n_cal"]
        chain = {n: {"fid": np.array(v)} for n, v in d["assignment_fidelity"].items()}
        plot(d["results"], chain, args)
        return
    chain, phys = build_chain(args)
    results = run(args, chain)
    os.makedirs(os.path.join(args.results_dir, "sweeps"), exist_ok=True)
    with open(os.path.join(args.results_dir, "sweeps", "readout_tomography.json"), "w") as f:
        json.dump({"shots": args.shots, "n_states": args.n_states, "physics": phys, "n_cal": args.n_cal,
                   "assignment_fidelity": {n: c["fid"].tolist() for n, c in chain.items()},
                   "C_true": {n: c["C_true"].tolist() for n, c in chain.items()},
                   "results": results}, f)
    plot(results, chain, args)


if __name__ == "__main__":
    main()