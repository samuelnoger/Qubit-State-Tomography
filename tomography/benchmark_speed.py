# tomography/benchmark_speed.py
"""Speed (and convergence) of MLE vs. the variable-shot neural reconstruction.

    python -m tomography.benchmark_speed --ensemble near_bell --checkpoint checkpoints/<your_near_bell_model>.pth \
        --shots 100 1000 10000 --n-states 100

MLE is timed for several iteration counts (tol=0, so every iteration runs) together with the infidelity it reaches,
which shows how many iterations it needs at each shot count. Without --checkpoint only MLE is timed.

Caveat: this MLE is a plain per-state numpy loop. A batched implementation (GPU or vectorized) would be much faster,
so the speedup printed here is an upper bound on what a network would save over a well-written MLE.
"""
import argparse
import importlib
import time

import numpy as np


def _load(module):
    """Import <module> from the 'utils' package if it exists there, otherwise from 'tomography'."""
    for pkg in ("utils", "tomography"):
        try:
            return importlib.import_module(f"{pkg}.{module}")
        except ImportError:
            continue
    raise ImportError(f"could not import '{module}' from utils or tomography")


_states, _ensembles, _reconstruct = _load("states"), _load("ensembles"), _load("reconstruct")
measurement_operators, sample_counts, fidelity = _states.measurement_operators, _states.sample_counts, _states.fidelity
mle = _reconstruct.mle
ENSEMBLES = {"broad": _ensembles.random_broad_state, "near_bell": _ensembles.random_near_bell_state}


def load_network(args):
    """Return predict(x, n_shots) -> (n, 4, 4) complex array, or None if no checkpoint is given.

    x holds the 36 frequencies (n, 36); the log10(N) column that the model expects is appended here.
    """
    if args.checkpoint is None:
        return None
    import torch
    from model.tomography_model import NeuralTomography

    ck = torch.load(args.checkpoint, map_location="cpu")
    wrapped = isinstance(ck, dict) and "state_dict" in ck
    hidden = ck.get("hidden_dim", args.hidden_dim) if wrapped else args.hidden_dim
    n_layers = ck.get("n_layers", args.n_layers) if wrapped else args.n_layers
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
    p.add_argument("--checkpoint", default=None, help="trained variable-shot network (.pth); omit to time MLE only")
    p.add_argument("--hidden-dim", type=int, default=256, help="used if the checkpoint does not store it")
    p.add_argument("--n-layers", type=int, default=3, help="used if the checkpoint does not store it")
    p.add_argument("--shots", type=int, nargs="+", default=[100, 1000, 10000])
    p.add_argument("--n-states", type=int, default=100)
    p.add_argument("--iters", type=int, nargs="+", default=[50, 200, 1000, 2000])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    E = measurement_operators()
    predict = load_network(args)
    if predict is None:
        print("(no --checkpoint given: timing MLE only)")

    for n_shots in args.shots:
        states = [ENSEMBLES[args.ensemble](rng) for _ in range(args.n_states)]
        counts = [sample_counts(s, E, n_shots, rng) for s in states]
        print(f"\n{args.ensemble} states, {n_shots} shots/setting, {args.n_states} states")
        t_mle = None
        for iters in args.iters:
            t0 = time.perf_counter()
            est = [mle(c, E, iters=iters, tol=0) for c in counts]
            t_mle = (time.perf_counter() - t0) / len(counts)
            print(f"  MLE, {iters:>5d} iterations : {t_mle * 1e3:8.2f} ms/state   infidelity {mean_infidelity(est, states):.3e}")

        if predict is not None:
            x = np.array([(c / n_shots).reshape(-1) for c in counts], dtype=np.float32)
            predict(x[:8], n_shots)                                         # warm-up
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
            print(f"  network, batch of {len(x)}      : {t_batch * 1e6:8.1f} us/state   infidelity {mean_infidelity(out, states):.3e}")
            print(f"  network, one state at a time : {t_single * 1e6:8.1f} us/state")
            print(f"  speedup vs. MLE ({args.iters[-1]} it.)    : x{t_mle / t_batch:,.0f} (batched), x{t_mle / t_single:,.0f} (single)")


if __name__ == "__main__":
    main()