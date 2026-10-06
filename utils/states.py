# tomography/states.py
"""Two-qubit states, Pauli measurement settings, shot sampling and fidelity (numpy only).

Conventions
-----------
* Pauli index: 0 = X, 1 = Y, 2 = Z.
* A measurement *setting* k = 3*i + j means "measure qubit 1 in Pauli i and qubit 2 in Pauli j" (9 settings).
* Outcomes m = 0..3 correspond to (s1, s2) = (+,+), (+,-), (-,+), (-,-).
* POVM element E[k, m] = P_i(s1) kron P_j(s2), with P_b(s) = (I + s*sigma_b)/2.
* Probabilities: p[k, m] = Tr(rho E[k, m]).
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
PAULI = [
    np.array([[0, 1], [1, 0]], dtype=complex),        # X
    np.array([[0, -1j], [1j, 0]], dtype=complex),     # Y
    np.array([[1, 0], [0, -1]], dtype=complex),       # Z
]
OUTCOMES = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
S1 = np.array([o[0] for o in OUTCOMES], dtype=float)
S2 = np.array([o[1] for o in OUTCOMES], dtype=float)


def measurement_operators():
    """Returns E with shape (9, 4, 4, 4): [setting, outcome, row, col]."""
    def proj(b, s):
        return (I2 + s * PAULI[b]) / 2
    E = np.zeros((9, 4, 4, 4), dtype=complex)
    for i in range(3):
        for j in range(3):
            for m, (s1, s2) in enumerate(OUTCOMES):
                E[3 * i + j, m] = np.kron(proj(i, s1), proj(j, s2))
    return E


# ---------------------------------------------------------------- states
def random_pure_state(rng):
    """Haar-random pure two-qubit state, returned as a density matrix."""
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def random_mixed_state(rng, rank=4):
    """Ginibre ensemble: rho = G G^dagger / Tr, with G of shape (4, rank). rank=1 gives a pure state."""
    G = rng.normal(size=(4, rank)) + 1j * rng.normal(size=(4, rank))
    rho = G @ G.conj().T
    return rho / np.trace(rho).real


def bell_state(name="phi+"):
    v = {"phi+": [1, 0, 0, 1], "phi-": [1, 0, 0, -1],
         "psi+": [0, 1, 1, 0], "psi-": [0, 1, -1, 0]}[name]
    psi = np.array(v, dtype=complex) / np.sqrt(2)
    return np.outer(psi, psi.conj())


# ---------------------------------------------------------------- measurement
def born_probabilities(rho, E):
    """p[k, m] = Tr(rho E[k, m]); each row sums to 1."""
    return np.einsum('kmij,ji->km', E, rho).real


def sample_counts(rho, E, n_shots, rng):
    """Multinomial counts, shape (9, 4), with n_shots shots in each setting."""
    p = np.clip(born_probabilities(rho, E), 0, None)
    p /= p.sum(axis=1, keepdims=True)
    return np.stack([rng.multinomial(n_shots, p[k]) for k in range(p.shape[0])])


# ---------------------------------------------------------------- metrics
def _psd_sqrt(a):
    w, v = np.linalg.eigh((a + a.conj().T) / 2)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.conj().T


def fidelity(rho, sigma):
    """Uhlmann fidelity F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2, in [0, 1]. Both must be physical."""
    s = _psd_sqrt(rho)
    m = s @ sigma @ s
    w = np.linalg.eigvalsh((m + m.conj().T) / 2)
    return float(min(np.sum(np.sqrt(np.clip(w, 0, None))) ** 2, 1.0))   # clip rounding error above 1


def purity(rho):
    return float(np.trace(rho @ rho).real)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    E = measurement_operators()

    # 1. States are valid density matrices
    for name, rho in [("pure", random_pure_state(rng)), ("mixed", random_mixed_state(rng)), ("bell", bell_state())]:
        ev = np.linalg.eigvalsh(rho)
        print(f"{name:>5s}: trace={np.trace(rho).real:.6f}  min eig={ev.min():.2e}  "
              f"hermitian={np.allclose(rho, rho.conj().T)}  purity={purity(rho):.4f}")

    # 2. Born probabilities sum to 1 in every setting; Bell state correlations
    p = born_probabilities(random_mixed_state(rng), E)
    print("row sums of p (should all be 1):", np.round(p.sum(axis=1), 12))
    pb = born_probabilities(bell_state("phi+"), E)
    print("Bell phi+, setting ZZ (k=8) probabilities (expect 0.5, 0, 0, 0.5):", np.round(pb[8], 6))
    print("Bell phi+, setting XX (k=0) probabilities (expect 0.5, 0, 0, 0.5):", np.round(pb[0], 6))

    # 3. Shot sampling
    c = sample_counts(bell_state("phi+"), E, 1000, rng)
    print("counts for ZZ with 1000 shots:", c[8])

    # 4. Fidelity: identical states -> 1, orthogonal pure states -> 0; compare with QuTiP
    a, b = random_pure_state(rng), random_pure_state(rng)
    print("F(a, a) =", round(fidelity(a, a), 10), "  F(phi+, psi-) =", round(fidelity(bell_state('phi+'), bell_state('psi-')), 10))
    m1, m2 = random_mixed_state(rng), random_mixed_state(rng)
    print("F(m1, m2) =", fidelity(m1, m2))
    try:
        import qutip as qt
        print("QuTiP check (fidelity is squared here; QuTiP returns the square root):",
              qt.fidelity(qt.Qobj(m1), qt.Qobj(m2)) ** 2)
    except ImportError:
        print("QuTiP not installed, skipping the cross-check")