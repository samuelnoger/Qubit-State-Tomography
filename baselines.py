# baselines.py
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier


def fidelity_per_qubit(pred, y):
    """Assignment fidelity 1 - (P(0|1) + P(1|0)) / 2 for each label column. Returns shape (n_qubits,)."""
    pred = np.asarray(pred).reshape(len(pred), -1)
    y = np.asarray(y).reshape(len(y), -1)
    out = []
    for i in range(y.shape[1]):
        p, t = pred[:, i], y[:, i]
        out.append(1.0 - 0.5 * (np.mean(p[t == 1] == 0) + np.mean(p[t == 0] == 1)))
    return np.array(out)


def assignment_fidelity(pred, y):
    """Mean assignment fidelity over qubits (a single number)."""
    return float(fidelity_per_qubit(pred, y).mean())


def _best_threshold(score, y):
    cands = np.quantile(score, np.linspace(0.01, 0.99, 400))
    fids = [assignment_fidelity(score > c, y) for c in cands]
    return cands[int(np.argmax(fids))]


def _fit_score(x, y, kind):
    """Return a function mapping records (N,T,C) -> scalar score, fitted on train data (y is 1D binary)."""
    if kind == "integrated":
        integ = x.sum(1)                                    # (N, C): integrate each channel
        d = integ[y == 1].mean(0) - integ[y == 0].mean(0)   # axis separating the two states
        return lambda z: z.sum(1) @ d
    if kind == "matched":
        kernel = x[y == 1].mean(0) - x[y == 0].mean(0)      # (T, C) time-resolved weights
        return lambda z: (z * kernel).sum((1, 2))
    if kind in ("lda", "lda_indep"):                        # best linear classifier on the given channels
        flat = x.reshape(len(x), -1)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(flat, y)
        return lambda z: clf.decision_function(z.reshape(len(z), -1))
    if kind == "gbm":                                       # nonlinear classical model, no temporal structure built in
        flat = x.reshape(len(x), -1)
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(flat, y)
        return lambda z: clf.decision_function(z.reshape(len(z), -1))
    raise ValueError(kind)


# Which methods only see the qubit's own channels ("independent") and which see everything ("joint")
OWN_CHANNELS = {"integrated", "matched", "lda_indep"}
LABELS = {
    "integrated": "integrated threshold",
    "matched":    "matched filter",
    "lda_indep":  "linear (LDA, own channels)",
    "lda":        "linear (LDA)",
    "gbm":        "gradient boosting",
}


def default_kinds(n_qubits):
    kinds = ["integrated", "matched"]
    if n_qubits > 1:
        kinds.append("lda_indep")           # isolates the benefit of seeing the other qubit's channels
    return kinds + ["lda", "gbm"]


def evaluate_baselines(train, test, kinds=None):
    """Returns {kind: per-qubit fidelity array}. Accepts torch tensors or numpy arrays."""
    xt, xe = np.asarray(train["x"]), np.asarray(test["x"])
    yt = np.asarray(train["y"]).reshape(len(xt), -1)
    ye = np.asarray(test["y"]).reshape(len(xe), -1)
    k = yt.shape[1]
    per_q = xt.shape[2] // k                # channels per qubit (2: I and Q)
    kinds = kinds or default_kinds(k)

    out = {}
    for kind in kinds:
        fids = []
        for i in range(k):
            sl = slice(per_q * i, per_q * (i + 1)) if kind in OWN_CHANNELS else slice(None)
            f = _fit_score(xt[:, :, sl], yt[:, i], kind)
            thr = _best_threshold(f(xt[:, :, sl]), yt[:, i])
            fids.append(fidelity_per_qubit(f(xe[:, :, sl]) > thr, ye[:, i])[0])
        out[kind] = np.array(fids)
    return out


def format_line(label, fids):
    """One printed line: mean first (so simple regexes can parse it), per-qubit values after."""
    line = f"  {label:<28s}: {np.mean(fids):.4f}"
    if len(fids) > 1:
        line += "   (" + ", ".join(f"q{i + 1} {v:.4f}" for i, v in enumerate(fids)) + ")"
    return line


if __name__ == "__main__":
    import torch
    data = torch.load("data/readout_dataset.pt")
    res = evaluate_baselines(data["train"], data["test"])
    for kind, fids in res.items():
        print(format_line(LABELS[kind], fids))