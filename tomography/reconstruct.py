# tomography/reconstruct.py
"""State reconstruction from measured counts: linear inversion, physical projection, and MLE."""
import numpy as np

from utils.states import PAULI, I2, S1, S2


# tomography/reconstruct.py
"""State reconstruction from measured counts: linear inversion, physical projection, and MLE."""
import numpy as np
import scipy.optimize as opt

from utils.states import PAULI, I2, S1, S2, bell_state


def rot_1q(a, b, c):
    """Standard ZYZ Euler decomposition for single-qubit rotation."""
    return np.array([
        [np.exp(-1j*(a+c)/2)*np.cos(b/2), -np.exp(-1j*(a-c)/2)*np.sin(b/2)],
        [np.exp( 1j*(a-c)/2)*np.sin(b/2),  np.exp( 1j*(a+c)/2)*np.cos(b/2)]
    ])


def parametric_near_bell(params):
    """Generates a 7-parameter state: local SU(2) rotations on a Bell state + depolarizing noise."""
    angles, mu = params[:6], params[6]
    U = np.kron(rot_1q(*angles[:3]), rot_1q(*angles[3:]))
    phi_plus = bell_state("phi+")
    return (1 - mu) * (U @ phi_plus @ U.conj().T) + (mu / 4.0) * np.eye(4)


def parametric_mle(counts, E_eff):
    """Model-based MLE for the 7-parameter near-Bell manifold."""
    def nll(params):
        rho = parametric_near_bell(params)
        p = np.maximum(np.einsum('kmij,ji->km', E_eff, rho).real, 1e-12)
        return -np.sum(counts * np.log(p))
    
    init = np.zeros(7)
    init[6] = 0.01  # Initialize mu at 1% to stay strictly inside the (0, 1) bound
    bounds = [(-np.pi, np.pi)] * 6 + [(0, 1)]
    
    res = opt.minimize(nll, init, bounds=bounds, method='L-BFGS-B')
    return parametric_near_bell(res.x)


def linear_inversion(counts):
    """rho = 1/4 [ II + sum_i <s_i I> s_i I + sum_j <I s_j> I s_j + sum_ij <s_i s_j> s_i s_j ].

    Single-qubit expectations are estimated from the marginals of the three settings that contain them
    and averaged. The result is Hermitian with trace 1 but may have negative eigenvalues.
    """
    freq = counts / counts.sum(axis=1, keepdims=True)          # (9, 4) outcome frequencies per setting
    rho = np.eye(4, dtype=complex) / 4
    a, b = np.zeros(3), np.zeros(3)
    for i in range(3):
        for j in range(3):
            k = 3 * i + j
            rho += (freq[k] @ (S1 * S2)) * np.kron(PAULI[i], PAULI[j]) / 4   # two-body correlator
            a[i] += freq[k] @ S1 / 3                                           # <sigma_i on qubit 1>
            b[j] += freq[k] @ S2 / 3                                           # <sigma_j on qubit 2>
    for i in range(3):
        rho += a[i] * np.kron(PAULI[i], I2) / 4
        rho += b[i] * np.kron(I2, PAULI[i]) / 4
    return rho


def project_to_physical(rho):
    """Closest physical state (Frobenius norm) to a Hermitian, unit-trace matrix (Smolin et al. 2012)."""
    rho = (rho + rho.conj().T) / 2
    vals, vecs = np.linalg.eigh(rho)
    vals, vecs = vals[::-1], vecs[:, ::-1]                      # descending
    d = len(vals)
    mu = np.zeros(d)
    i, acc = d, 0.0
    while i > 0 and vals[i - 1] + acc / i < 0:                   # zero out negative eigenvalues,
        acc += vals[i - 1]                                       # spreading their weight over the rest
        i -= 1
    mu[:i] = vals[:i] + acc / i
    return (vecs * mu) @ vecs.conj().T


def log_likelihood(rho, counts, E):
    p = np.maximum(np.einsum('kmij,ji->km', E, rho).real, 1e-12)
    return float(np.sum(counts * np.log(p)))


def mle(counts, E, iters=500, tol=1e-10):
    """Maximum likelihood via the R rho R iteration, starting from the maximally mixed state."""
    f = counts / counts.sum()                                    # frequencies, summing to 1 over all outcomes
    rho = np.eye(4, dtype=complex) / 4
    for _ in range(iters):
        p = np.maximum(np.einsum('kmij,ji->km', E, rho).real, 1e-12)
        R = np.einsum('km,kmij->ij', f / p, E)
        new = R @ rho @ R
        new /= np.trace(new).real
        done = np.linalg.norm(new - rho) < tol
        rho = new
        if done:
            break
    return rho


if __name__ == "__main__":
    from utils.states import (measurement_operators, random_pure_state, random_mixed_state,
                                   bell_state, sample_counts, born_probabilities, fidelity)
    rng = np.random.default_rng(1)
    E = measurement_operators()

    # 1. With exact probabilities (infinite shots), linear inversion recovers rho exactly
    rho = random_mixed_state(rng)
    exact = born_probabilities(rho, E)
    print("linear inversion, exact probabilities, max error:", np.abs(linear_inversion(exact) - rho).max())

    # 2. Projection leaves physical states unchanged, and fixes unphysical ones
    print("projection of a physical state, max change:", np.abs(project_to_physical(rho) - rho).max())
    noisy = linear_inversion(sample_counts(random_pure_state(rng), E, 20, rng))
    print("min eigenvalue before / after projection:",
          np.linalg.eigvalsh(noisy).min().round(4), "/", np.linalg.eigvalsh(project_to_physical(noisy)).min().round(6))

    # 3. MLE with exact probabilities recovers a full-rank state; the likelihood never decreases
    est = mle(exact * 1e6, E, iters=3000)
    print("MLE, exact probabilities, max error:", np.abs(est - rho).max())
    counts = sample_counts(rho, E, 200, rng)
    lls = []
    r = np.eye(4, dtype=complex) / 4
    for n in (1, 10, 50, 200):
        lls.append(log_likelihood(mle(counts, E, iters=n, tol=0), counts, E))
    print("log-likelihood after 1, 10, 50, 200 iterations (should increase):", np.round(lls, 3))

    # 4. Finite shots: both estimators approach the truth as shots grow
    for n in (30, 300, 3000, 30000):
        c = sample_counts(rho, E, n, rng)
        f_li = fidelity(project_to_physical(linear_inversion(c)), rho)
        f_ml = fidelity(mle(c, E), rho)
        print(f"{n:>6d} shots/setting: 1-F  linear+projection = {1 - f_li:.2e}   MLE = {1 - f_ml:.2e}")