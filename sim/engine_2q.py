# sim/engine_2q.py
"""Two-qubit multiplexed readout (units: microseconds).

Reuses the single-qubit pieces from sim.engine:
  - qubit_states: QuTiP mcsolve jump trajectories (T1 decay), one call per qubit.
    The two decays are independent, so no 4x4 Hilbert space is needed.

Crosstalk, two separate effects:
  zeta : cross-dispersive shift. Resonator i sees chi*s_i + zeta*s_j (nonlinear in the records).
  leak : linear signal leakage. Channel i measures alpha_i + leak * alpha_j.

A static J*sz*sz coupling is deliberately omitted: it is diagonal in the computational
basis and has no effect on records generated from basis-state preparations.
"""
import numpy as np
from sim.engine import qubit_states, cavity_response

COMBOS = [(0, 0), (0, 1), (1, 0), (1, 1)]          # (qubit 1 excited?, qubit 2 excited?)


def cavity_response_2q(s1, s2, tlist, kappa, chi, zeta, eps):
    """Noise-free complex amplitudes of the two resonators (before leakage)."""
    dt = tlist[1] - tlist[0]
    a1 = np.zeros(s1.shape[0], dtype=complex)
    a2 = np.zeros(s1.shape[0], dtype=complex)
    out1 = np.zeros(s1.shape, dtype=complex)
    out2 = np.zeros(s1.shape, dtype=complex)
    for k in range(1, len(tlist)):
        lam1 = kappa / 2 + 1j * (chi * s1[:, k - 1] + zeta * s2[:, k - 1])
        lam2 = kappa / 2 + 1j * (chi * s2[:, k - 1] + zeta * s1[:, k - 1])
        ss1, ss2 = -1j * eps / lam1, -1j * eps / lam2
        a1 = ss1 + (a1 - ss1) * np.exp(-lam1 * dt)
        a2 = ss2 + (a2 - ss2) * np.exp(-lam2 * dt)
        out1[:, k], out2[:, k] = a1, a2
    return out1, out2


def simulate_readout_2q(n_per_class, T1=3.0, t_ro=2.0, dt=0.02, kappa=10.0, chi=5.0,
                        zeta=1.0, leak=0.1, eps=5.0, sigma=3.0, seed=0):
    """Returns tlist, records (N, T, 4) = [I1, Q1, I2, Q2], labels (N, 2) = [q1, q2] in {0, 1}."""
    rng = np.random.default_rng(seed)
    tlist = np.arange(0, t_ro + 1e-9, dt)
    gamma = 1.0 / T1

    recs, labels = [], []
    for q1, q2 in COMBOS:
        s1 = qubit_states(bool(q1), gamma, tlist, n_per_class, seed=int(rng.integers(2**31)))
        s2 = qubit_states(bool(q2), gamma, tlist, n_per_class, seed=int(rng.integers(2**31)))
        a1, a2 = cavity_response_2q(s1, s2, tlist, kappa, chi, zeta, eps)
        m1, m2 = a1 + leak * a2, a2 + leak * a1                   # linear leakage between channels
        x = np.stack([m1.real, m1.imag, m2.real, m2.imag], axis=-1)
        x = x + rng.normal(0, sigma, size=x.shape)
        recs.append(x)
        labels.append(np.tile([q1, q2], (n_per_class, 1)))

    X = np.concatenate(recs).astype(np.float32)
    y = np.concatenate(labels).astype(np.float32)
    perm = rng.permutation(len(y))
    return tlist, X[perm], y[perm]


if __name__ == "__main__":
    T1, tl = 3.0, np.linspace(0, 2.0, 101)

    # Check 1: with no crosstalk, resonator 1 reproduces the verified single-qubit response
    rng = np.random.default_rng(0)
    s1 = qubit_states(True, 1 / T1, tl, 200, seed=1)
    s2 = qubit_states(True, 1 / T1, tl, 200, seed=2)
    a1, _ = cavity_response_2q(s1, s2, tl, 10.0, 5.0, 0.0, 5.0)
    ref = cavity_response(s1, tl, 10.0, 5.0, 5.0)
    print("zeta=0 vs single-qubit engine, max diff:", np.abs(a1 - ref).max())      # ~1e-12

    # Check 2: each qubit decays as exp(-t/T1), and the two decays are independent
    s1 = qubit_states(True, 1 / T1, tl, 4000, seed=3)
    s2 = qubit_states(True, 1 / T1, tl, 4000, seed=4)
    for name, s in (("qubit 1", s1), ("qubit 2", s2)):
        print(f"{name}: max |P_e - exp(-t/T1)| =", np.abs((s + 1).mean(0) / 2 - np.exp(-tl / T1)).max())
    d1, d2 = s1[:, -1] > 0, s2[:, -1] > 0                      # still excited at the end?
    print("decay-event correlation (should be ~0):", np.corrcoef(d1, d2)[0, 1])

    # Check 3: steady state with the cross-dispersive shift (s1=+1, s2=-1), analytic vs simulated
    long_t = np.linspace(0, 5.0, 251)
    a1, a2 = cavity_response_2q(np.ones((1, 251)), -np.ones((1, 251)), long_t, 10.0, 5.0, 1.0, 5.0)
    print("res.1 sim:", a1[0, -1], " analytic:", -1j * 5.0 / (5.0 + 1j * (5.0 - 1.0)))
    print("res.2 sim:", a2[0, -1], " analytic:", -1j * 5.0 / (5.0 + 1j * (-5.0 + 1.0)))

    # Shapes
    tl2, X, y = simulate_readout_2q(n_per_class=100)
    print("records:", X.shape, " labels:", y.shape, " mean labels:", y.mean(0))