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

    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    print(parse_args())