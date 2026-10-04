# /train/arguments.py
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Two-qubit state tomography")

    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--results-dir', type=str, default='results/', help='Root folder for figures and result files')

    # Ideal tomography experiment
    parser.add_argument('--shots', type=int, nargs='+', default=[10, 30, 100, 300, 1000, 3000, 10000],
                        help='Shots per measurement setting (there are 9 settings)')
    parser.add_argument('--n-states', type=int, default=50, help='Random states per state type and shot number')
    parser.add_argument('--state-types', type=str, nargs='+', default=['pure', 'mixed', 'bell'],
                        choices=['pure', 'mixed', 'bell'], help='Which kinds of states to reconstruct')
    parser.add_argument('--mixed-rank', type=int, default=4, choices=[1, 2, 3, 4], help='Rank of the random mixed states')
    parser.add_argument('--mle-iters', type=int, default=500, help='Maximum iterations of the MLE algorithm')

    # Milestone 2: readout chain
    parser.add_argument('--replot', action='store_true', help='Redraw the milestone 2 figure from the saved JSON, no simulation')
    parser.add_argument('--classifiers', type=str, nargs='+', default=['matched', 'lda', 'cnn'],
                        choices=['integrated', 'matched', 'lda_indep', 'lda', 'cnn'], help='Readout classifiers to compare')
    parser.add_argument('--cnn-checkpoint', type=str, default='checkpoints/readout_cnn_2q.pth',
                        help='Two-qubit CNN checkpoint from the readout project (its physics settings are used)')
    parser.add_argument('--n-train-readout', type=int, default=5000, help='Training records per joint state for the linear classifiers')
    parser.add_argument('--n-cal', type=int, default=1000, help='Calibration records per joint state (estimates the confusion matrix)')
    parser.add_argument('--n-cal-true', type=int, default=10000, help='Records per joint state defining the "true" readout channel')
    # Readout physics, used only if no CNN checkpoint supplies it (units: microseconds)
    parser.add_argument('--T1', type=float, default=3.0)
    parser.add_argument('--t-ro', type=float, default=2.0)
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--kappa', type=float, default=10.0)
    parser.add_argument('--chi', type=float, default=5.0)
    parser.add_argument('--eps', type=float, default=5.0)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--zeta', type=float, default=1.0)
    parser.add_argument('--leak', type=float, default=0.1)

    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    print(parse_args())