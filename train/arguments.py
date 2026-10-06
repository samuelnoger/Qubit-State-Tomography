# train/arguments.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Two-qubit state tomography")

    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--results-dir', type=str, default='results/', help='Root folder for figures and result files')
    parser.add_argument('--device', type=str, default='mps', help='Compute device (cpu, cuda, mps)')

    # Ideal tomography experiment (Milestone 1)
    parser.add_argument('--shots', type=int, nargs='+', default=[10, 30, 100, 300, 1000, 3000, 10000], help='Shots per measurement setting')
    parser.add_argument('--n-states', type=int, default=50, help='Random states per state type and shot number')
    parser.add_argument('--state-types', type=str, nargs='+', default=['pure', 'mixed', 'bell'], choices=['pure', 'mixed', 'bell'])
    parser.add_argument('--mixed-rank', type=int, default=4, choices=[1, 2, 3, 4], help='Rank of the random mixed states')
    parser.add_argument('--mle-iters', type=int, default=500, help='Maximum iterations of the MLE algorithm')

    # Readout chain (Milestone 2)
    parser.add_argument('--replot', action='store_true', help='Redraw the milestone 2 figure from the saved JSON')
    parser.add_argument('--classifiers', type=str, nargs='+', default=['matched', 'lda', 'cnn'], choices=['integrated', 'matched', 'lda_indep', 'lda', 'cnn'])
    parser.add_argument('--cnn-checkpoint', type=str, default='checkpoints/readout_cnn_2q.pth')
    parser.add_argument('--n-train-readout', type=int, default=5000)
    parser.add_argument('--n-cal', type=int, default=1000)
    parser.add_argument('--n-cal-true', type=int, default=10000)
    
    # Readout physics
    parser.add_argument('--T1', type=float, default=3.0)
    parser.add_argument('--t-ro', type=float, default=2.0)
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--kappa', type=float, default=10.0)
    parser.add_argument('--chi', type=float, default=5.0)
    parser.add_argument('--eps', type=float, default=5.0)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--zeta', type=float, default=1.0)
    parser.add_argument('--leak', type=float, default=0.1)

    # General Training Arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    
    # Neural Tomography (Milestone 3)
    parser.add_argument('--tomo-data-path', type=str, default='data/neural_tomo_data.pt')
    parser.add_argument('--tomo-state-type', type=str, choices=['pure', 'mixed', 'bell', 'broad', 'near_bell'], default='pure', help='State family for neural tomography')
    parser.add_argument('--tomo-shots', type=int, default=1000, help='Shots per setting for the neural dataset')
    parser.add_argument('--n-train-tomo', type=int, default=50000)
    parser.add_argument('--n-val-tomo', type=int, default=5000)
    parser.add_argument('--n-test-tomo', type=int, default=10000)
    parser.add_argument('--bell-noise', type=float, default=0.05, help='Depolarising noise added to exact Bell states')
    parser.add_argument('--tomo-hidden-dim', type=int, default=256)
    parser.add_argument('--tomo-n-layers', type=int, default=3, help='Number of hidden layers in the neural tomography model')

    args, unknown = parser.parse_known_args()
    return args