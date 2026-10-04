# Two-Qubit State Tomography Through a Noisy Readout Chain

A self-contained project on **quantum state tomography**: reconstructing the density matrix of two qubits from measurement statistics, first with ideal readout, then through the simulated readout chain of a superconducting qubit. The readout classifiers (matched filter, linear discriminant, and a 1D CNN) come from [Qubit-Readout-ML](https://github.com/samuelnoger/Qubit-Readout-ML).

<img src="results/figures/ideal_tomography.png" alt="Ideal tomography" width="600">
<img src="results/figures/readout_tomography.png" alt="Tomography through the readout chain" width="600">

---

## Background

A two-qubit density matrix $\rho$ has 15 real parameters. Measuring each qubit in the $X$, $Y$ or $Z$ basis yields 9 measurement settings with four outcomes each. The probability of an outcome is given by the Born rule. Reconstruction from finite shots is a statistical estimate, evaluated by the infidelity $1 - F$ against the true state.

## Methods

**Reconstruction:**
- *Linear inversion with projection:* Estimates Pauli correlators from counts, projects the result onto the nearest physical state.
- *Maximum likelihood (MLE):* Iterative solver ensuring the estimate remains physical.

**Readout chain:** The QuTiP simulator implements $T_1$ decay, dispersive cavity response, and multi-qubit crosstalk. A classifier predicts the bit string. The chain is summarized by a $4 \times 4$ confusion matrix $C$.

**Reconstructions through the chain:**
- *Naive:* Treats reported counts as if readout were perfect.
- *Corrected:* Runs MLE with effective measurement operators, using a confusion matrix calibrated from a separate dataset.

## Results

### Ideal readout
- **Bell states:** MLE falls as $1/N$ and reaches roughly $10^{-6}$ at $3 \times 10^5$ shots. Linear inversion falls as $1/\sqrt{N}$.
- **Random pure states:** Both estimators scale roughly as $1/\sqrt{N}$.
- **Random mixed states:** Both estimators scale roughly as $1/N$ at high shot counts.

### Through the readout chain
- **Ignoring readout errors gives a hard noise floor:** Without correction, infidelity plateaus. More shots do not improve the reconstruction.
- **A calibrated correction recovers scaling:** With MLE correction, the infidelity resumes falling with the number of shots. For Bell states, a significant gap remains relative to the ideal readout curve.
- **Classifier impact:** Uncorrected, the matched filter performs worse than the LDA or CNN. With correction, the CNN and LDA perform similarly and slightly outperform the matched filter.

## Development Methodology

The core CNN architecture, QuTiP simulation boilerplate, and classical baselines were scaffolded with the assistance of AI coding tools. Primary technical contributions focus on structuring the quantum state tomography math (Maximum Likelihood Estimation and Linear Inversion), designing the physical simulation to isolate multi-qubit crosstalk, optimizing hardware utilization for Apple Silicon (MPS), and decoupling the analog data generation pools to bypass computational bottlenecks during high-shot evaluation.

## Quick start
```bash
pip install -r requirements.txt

python -m tomography.states
python -m tomography.reconstruct
python -m tomography.ideal_tomography --shots 30 300 3000 30000 300000 --mle-iters 2000

python -m tomography.readout_tomography \
    --shots 100 300 1000 3000 10000 30000 100000 \
    --n-states 50 --n-cal 20000 --n-cal-true 40000 --mle-iters 2000
