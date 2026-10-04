# Two-Qubit State Tomography Through a Noisy Readout Chain

A self-contained project on **quantum state tomography**: reconstructing the density matrix of two qubits from measurement statistics, first with ideal readout, then through the simulated readout chain of a superconducting qubit. The readout classifiers (matched filter, linear discriminant, and a 1D CNN) come from [Qubit-Readout-ML](https://github.com/samuelnoger/Qubit-Readout-ML).

<img src="results/figures/ideal_tomography.png" alt="Ideal tomography" width="400">
<img src="results/figures/readout_tomography.png" alt="Tomography through the readout chain" width="400">

---

## Background

A two-qubit density matrix $\rho$ has 15 real parameters[cite: 10]. Measuring each qubit in the $X$, $Y$ or $Z$ basis yields 9 measurement settings with four outcomes each[cite: 10]. The probability of an outcome is given by the Born rule[cite: 10]. Reconstruction from finite shots is a statistical estimate, evaluated by the infidelity $1 - F$ against the true state[cite: 10].

## Methods

**Reconstruction:**
- *Linear inversion with projection:* Estimates Pauli correlators from counts, projects the result onto the nearest physical state[cite: 10].
- *Maximum likelihood (MLE):* Iterative solver ensuring the estimate remains physical[cite: 10].

**Readout chain:** The QuTiP simulator implements $T_1$ decay, dispersive cavity response, and multi-qubit crosstalk[cite: 10]. A classifier predicts the bit string[cite: 10]. The chain is summarized by a $4 \times 4$ confusion matrix $C$[cite: 10].

**Reconstructions through the chain:**
- *Naive:* Treats reported counts as if readout were perfect[cite: 10].
- *Corrected:* Runs MLE with effective measurement operators, using a confusion matrix calibrated from a separate dataset[cite: 10].

## Results

### Ideal readout
- **Bell states:** MLE falls as $1/N$ and reaches roughly $10^{-6}$ at $3 \times 10^5$ shots[cite: 10]. Linear inversion falls as $1/\sqrt{N}$[cite: 10].
- **Random pure states:** Both estimators scale roughly as $1/\sqrt{N}$[cite: 10].
- **Random mixed states:** Both estimators scale roughly as $1/N$ at high shot counts[cite: 10].

### Through the readout chain
- **Ignoring readout errors gives a hard noise floor:** Without correction, infidelity plateaus[cite: 10]. More shots do not improve the reconstruction[cite: 9, 10].
- **A calibrated correction recovers scaling:** With MLE correction, the infidelity resumes falling with the number of shots[cite: 9, 10]. For Bell states, a significant gap remains relative to the ideal readout curve[cite: 9].
- **Classifier impact:** Uncorrected, the matched filter performs worse than the LDA or CNN[cite: 9, 10]. With correction, the CNN and LDA perform similarly and slightly outperform the matched filter[cite: 9, 10].

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