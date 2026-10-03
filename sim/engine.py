# sim/engine.py
"""Dispersive qubit readout simulator (units: microseconds).

Qubit: QuTiP mcsolve (quantum jumps, T1 decay) -> s(t) = +1 (excited) / -1 (ground).
Cavity: dα/dt = -i*eps - (kappa/2 + i*chi*s(t)) * α, solved exactly per time step.
Record: (I, Q) = (Re α, Im α) + Gaussian noise per sample.
"""
import numpy as np
import qutip as qt

_exc = qt.basis(2, 0)                      # QuTiP convention: |0> is "up" = excited
_gnd = qt.basis(2, 1)
_proj_e = _exc * _exc.dag()
_H0 = 0 * qt.sigmaz()


def qubit_states(excited, gamma, tlist, ntraj, seed=None):
    """Return s(t) in {+1, -1}, shape (ntraj, len(tlist))."""
    if not excited:                        # ground state never decays
        return -np.ones((ntraj, len(tlist)))
    res = qt.mcsolve(
        _H0, _exc, tlist,
        c_ops=[np.sqrt(gamma) * qt.sigmam()],
        e_ops=[_proj_e],
        ntraj=ntraj,
        options={"keep_runs_results": True, "progress_bar": False},
        seeds=seed,
    )
    p_e = np.array(res.runs_expect[0])     # (ntraj, nt), values 0/1 per trajectory
    return 2.0 * np.round(p_e) - 1.0


def cavity_response(s, tlist, kappa, chi, eps):
    """Noise-free complex cavity amplitude for each qubit trajectory s."""
    dt = tlist[1] - tlist[0]
    alpha = np.zeros(s.shape[0], dtype=complex)
    out = np.zeros(s.shape, dtype=complex)
    for k in range(1, len(tlist)):
        lam = kappa / 2 + 1j * chi * s[:, k - 1]    # qubit state during the step
        ss = -1j * eps / lam
        alpha = ss + (alpha - ss) * np.exp(-lam * dt)
        out[:, k] = alpha
    return out


def simulate_readout(n_per_class, T1=3.0, t_ro=2.0, dt=0.02,
                     kappa=10.0, chi=5.0, eps=5.0, sigma=3.0, seed=0):
    """Returns records (N, T, 2) float32 and labels (N,) (1 = prepared excited)."""
    rng = np.random.default_rng(seed)
    tlist = np.arange(0, t_ro + 1e-9, dt)
    gamma = 1.0 / T1

    recs, labels = [], []
    for lab in (0, 1):
        s = qubit_states(bool(lab), gamma, tlist, n_per_class, seed=int(rng.integers(2**31)))
        a = cavity_response(s, tlist, kappa, chi, eps)
        x = np.stack([a.real, a.imag], axis=-1)
        x = x + rng.normal(0, sigma, size=x.shape)
        recs.append(x)
        labels.append(np.full(n_per_class, lab))

    X = np.concatenate(recs).astype(np.float32)
    y = np.concatenate(labels).astype(np.float32)
    perm = rng.permutation(len(y))
    return tlist, X[perm], y[perm]


if __name__ == "__main__":
    # Sanity check 1: jump statistics must follow exp(-t/T1)
    T1, tl = 3.0, np.linspace(0, 2.0, 101)
    s = qubit_states(True, 1 / T1, tl, ntraj=4000, seed=1)
    p_e = (s + 1).mean(0) / 2
    print("max |P_e - exp(-t/T1)|:", np.abs(p_e - np.exp(-tl / T1)).max())  # ~ 1e-2 (shot noise)

    # Sanity check 2: noise-free steady state matches -i*eps/lambda
    kappa, chi, eps = 10.0, 5.0, 5.0
    long_t = np.linspace(0, 5.0, 251)
    a = cavity_response(np.ones((1, 251)), long_t, kappa, chi, eps)[0, -1]
    print("alpha_ss sim:", a, " analytic:", -1j * eps / (kappa / 2 + 1j * chi))