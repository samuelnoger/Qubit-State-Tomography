# utils/ensembles.py
"""Additional state ensembles for the neural reconstruction (numpy only).

broad     : rank drawn uniformly from {1, 2, 3, 4}, then a Ginibre state of that rank (rank 1 = Haar-random pure).
near_bell : a Bell pair (phi+) with small random local rotations and depolarizing noise, i.e. the kind of state
            one actually tries to prepare and then characterizes.
"""
import numpy as np

from utils.states import PAULI, bell_state, random_mixed_state


def random_broad_state(rng, ranks=(1, 2, 3, 4)):
    return random_mixed_state(rng, rank=int(rng.choice(ranks)))


def _local_unitary(rng, angle_std):
    """exp(-i theta.sigma / 2) with a random rotation vector theta ~ N(0, angle_std^2)."""
    theta = rng.normal(0, angle_std, size=3)
    gen = sum(t * P for t, P in zip(theta, PAULI)) / 2
    w, v = np.linalg.eigh(gen)
    return (v * np.exp(-1j * w)) @ v.conj().T


def random_near_bell_state(rng, p_max=0.2, angle_std=0.15):
    U = np.kron(_local_unitary(rng, angle_std), _local_unitary(rng, angle_std))
    p = rng.uniform(0, p_max)                                   # depolarizing noise
    return (1 - p) * U @ bell_state("phi+") @ U.conj().T + p * np.eye(4) / 4


if __name__ == "__main__":
    from utils.states import measurement_operators, sample_counts, fidelity, purity
    from tomography.reconstruct import linear_inversion, project_to_physical, mle

    rng = np.random.default_rng(0)
    E = measurement_operators()
    for name, make in (("broad", random_broad_state), ("near_bell", random_near_bell_state)):
        states = [make(rng) for _ in range(300)]
        ev = min(np.linalg.eigvalsh(s).min() for s in states)
        pur = [purity(s) for s in states]
        print(f"{name:>9s}: trace={np.mean([np.trace(s).real for s in states]):.6f}  min eigenvalue={ev:.1e}  "
              f"purity range [{min(pur):.2f}, {max(pur):.2f}]")

    states = [random_near_bell_state(rng) for _ in range(200)]
    phi = bell_state("phi+")
    li, ml = [], []
    for s in states:
        c = sample_counts(s, E, 1000, rng)
        li.append(1 - fidelity(project_to_physical(linear_inversion(c)), s))
        ml.append(1 - fidelity(mle(c, E, iters=1000), s))
    print("\nnear_bell family at 1000 shots/setting (200 states):")
    print(f"  constant guess 'ideal phi+'     : {np.mean([1 - fidelity(phi, s) for s in states]):.3e}")
    print(f"  linear inversion + projection   : {np.mean(li):.3e}")
    print(f"  maximum likelihood              : {np.mean(ml):.3e}")