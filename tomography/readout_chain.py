# tomography/readout_chain.py
"""Readout chain for tomography: analog records -> classifier -> reported bits -> 4x4 confusion matrix.

Bit convention (used everywhere):
    bit 0 = ground = Pauli outcome +1,   bit 1 = excited = Pauli outcome -1.
The tomography outcome index m = 0..3 for (s1, s2) = (+,+), (+,-), (-,+), (-,-) is therefore exactly
m = 2*b1 + b2, the same index as the joint label 2*q1 + q2 of the readout project.

Readout happens after the basis rotation, so one confusion matrix C[m_reported, t_true] applies to all
9 settings. Reported-outcome probabilities are p_reported = C @ p_true.
"""
import os

import numpy as np

from baselines import _fit_score, _best_threshold, OWN_CHANNELS, fidelity_per_qubit
from tomography.states import born_probabilities


def bit_index(bits):
    bits = np.asarray(bits).astype(int)
    return 2 * bits[:, 0] + bits[:, 1]


def confusion_from_predictions(pred_bits, true_bits):
    """C[m, t] = P(report outcome m | true outcome t). Columns sum to 1."""
    C = np.zeros((4, 4))
    np.add.at(C, (bit_index(pred_bits), bit_index(true_bits)), 1.0)
    return C / C.sum(axis=0, keepdims=True)


def effective_operators(E, C):
    """E'[k, m] = sum_t C[m, t] E[k, t]: POVM elements for *reported* outcomes (still sum to I per setting)."""
    return np.einsum('mt,ktij->kmij', C, E)


def sample_counts_readout(rho, E, C, n_shots, rng):
    """Counts of reported outcomes, shape (9, 4)."""
    p_rep = np.clip(born_probabilities(rho, E), 0, None) @ C.T
    p_rep = np.clip(p_rep, 0, None)
    p_rep /= p_rep.sum(axis=1, keepdims=True)
    return np.stack([rng.multinomial(n_shots, p_rep[k]) for k in range(p_rep.shape[0])])


# ---------------------------------------------------------------- classifiers -> predictors
def _make_predictor(fitted):
    def predict(x):
        return np.stack([(f(x[:, :, sl]) > thr) for sl, f, thr in fitted], axis=1).astype(int)
    return predict


def fit_linear_predictors(x, y, kinds):
    """Fit classical classifiers on records x (N, T, 4) with labels y (N, 2). Returns {kind: predict(x) -> (N, 2) bits}."""
    per_q = x.shape[2] // y.shape[1]
    predictors = {}
    for kind in kinds:
        fitted = []
        for i in range(y.shape[1]):
            sl = slice(per_q * i, per_q * (i + 1)) if kind in OWN_CHANNELS else slice(None)
            f = _fit_score(x[:, :, sl], y[:, i], kind)
            fitted.append((sl, f, _best_threshold(f(x[:, :, sl]), y[:, i])))
        predictors[kind] = _make_predictor(fitted)
    return predictors


def load_cnn_predictor(path):
    """Load the two-qubit CNN/GRU checkpoint from the readout project. Returns (predict, phys)."""
    import torch
    from model.readout_model import ReadoutCNN, ReadoutGRU
    ck = torch.load(path, map_location="cpu")
    if ck["n_qubits"] != 2:
        raise ValueError(f"Checkpoint has n_qubits={ck['n_qubits']}, expected 2")
    if ck["arch"] == "cnn":
        model = ReadoutCNN(ck["seq_len"], in_channels=ck["in_channels"], n_outputs=ck["n_qubits"], hidden=ck["hidden"])
    else:
        model = ReadoutGRU(in_channels=ck["in_channels"], n_outputs=ck["n_qubits"], hidden=ck["hidden"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    scale = ck["scale"]

    def predict(x):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 4096):
                xb = torch.from_numpy(np.ascontiguousarray(x[i:i + 4096])).float() / scale
                out.append((model(xb) > 0).numpy())
        return np.concatenate(out).astype(int)

    return predict, ck["phys"]


def physics_from_args(args):
    return dict(T1=args.T1, t_ro=args.t_ro, dt=args.dt, kappa=args.kappa, chi=args.chi,
                eps=args.eps, sigma=args.sigma, zeta=args.zeta, leak=args.leak)


def build_chain(args):
    """Simulate records, fit/load the classifiers, and tabulate their confusion matrices.

    Returns chain = {name: dict(C_true, C_cal, fid)} and the physics used.
      C_true : confusion matrix from a large simulated set, treated as the real readout channel.
      C_cal  : matrix from a small calibration set (args.n_cal records per state), what a lab would have.
    """
    from sim.engine_2q import simulate_readout_2q

    names, predictors, phys = list(args.classifiers), {}, None
    if "cnn" in names:
        if os.path.exists(args.cnn_checkpoint):
            predictors["cnn"], phys = load_cnn_predictor(args.cnn_checkpoint)
            print(f"Loaded CNN from {args.cnn_checkpoint}")
        else:
            print(f"No checkpoint at {args.cnn_checkpoint}; skipping the CNN")
            names.remove("cnn")
    if phys is None:
        phys = physics_from_args(args)
    print("Readout physics:", phys)

    linear = [n for n in names if n != "cnn"]
    if linear:
        _, x_tr, y_tr = simulate_readout_2q(args.n_train_readout, seed=args.seed + 11, **phys)
        predictors.update(fit_linear_predictors(x_tr, y_tr, linear))
    _, x_cal, y_cal = simulate_readout_2q(args.n_cal, seed=args.seed + 12, **phys)
    _, x_true, y_true = simulate_readout_2q(args.n_cal_true, seed=args.seed + 13, **phys)

    chain = {}
    for name in names:
        pred_true = predictors[name](x_true)
        chain[name] = dict(C_cal=confusion_from_predictions(predictors[name](x_cal), y_cal),
                           C_true=confusion_from_predictions(pred_true, y_true),
                           fid=fidelity_per_qubit(pred_true, y_true))
        print(f"\n{name}: assignment fidelity per qubit = {np.round(chain[name]['fid'], 4)}")
        print("confusion matrix C[reported, true] (rows: 00 01 10 11 reported):")
        print(np.round(chain[name]["C_true"], 3))
    return chain, phys