# utils/features.py
"""Pauli correlators as input features for the neural reconstruction (numpy only).

The 36 measured frequencies (9 settings x 4 outcomes, flattened as 4*k + m) are mapped linearly to the 15
Pauli expectation values <sigma_i x I>, <I x sigma_j>, <sigma_i x sigma_j>, i, j in (X, Y, Z). These are
centred around 0 and lie in [-1, 1], unlike the raw frequencies, which sit around 0.25.
Order of the 15 features: [XI, YI, ZI, IX, IY, IZ, XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ].
"""
import numpy as np

from utils.states import S1, S2


def feature_matrix():
    """M with shape (36, 15): features = freq.reshape(-1) @ M."""
    M = np.zeros((36, 15))
    for i in range(3):
        for j in range(3):
            k = 3 * i + j
            for m in range(4):
                M[4 * k + m, i] += S1[m] / 3                    # <sigma_i on qubit 1>, averaged over 3 partner bases
                M[4 * k + m, 3 + j] += S2[m] / 3                # <sigma_j on qubit 2>
                M[4 * k + m, 6 + 3 * i + j] += S1[m] * S2[m]    # two-body correlator
    return M


if __name__ == "__main__":
    from utils.states import measurement_operators, random_mixed_state, sample_counts, PAULI, I2
    from tomography.reconstruct import linear_inversion

    rng = np.random.default_rng(0)
    E, M = measurement_operators(), feature_matrix()
    rho = random_mixed_state(rng)
    counts = sample_counts(rho, E, 500, rng)
    f = M.T @ (counts / counts.sum(axis=1, keepdims=True)).reshape(-1)       # 15 features

    # rebuild rho from the features and compare with linear_inversion
    rec = np.eye(4, dtype=complex) / 4
    for i in range(3):
        rec += f[i] * np.kron(PAULI[i], I2) / 4 + f[3 + i] * np.kron(I2, PAULI[i]) / 4
        for j in range(3):
            rec += f[6 + 3 * i + j] * np.kron(PAULI[i], PAULI[j]) / 4
    print("rho from features vs linear_inversion, max difference:", np.abs(rec - linear_inversion(counts)).max())
    print("feature range:", f.min().round(3), f.max().round(3))