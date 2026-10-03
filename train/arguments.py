# /train/arguments.py
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Qubit readout classification: data generation and training")

    # 1. Hardware / Device settings
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Hardware device for training')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed')

    # 2. Training hyperparameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Initial learning rate for Adam')
    parser.add_argument('--min-lr', type=float, default=1e-5, help='Minimum learning rate for the cosine scheduler')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')

    # 3. Model architecture
    parser.add_argument('--arch', type=str, default='cnn', choices=['cnn', 'gru'], help='Classifier architecture')
    parser.add_argument('--hidden-neurons', type=int, default=64, help='Hidden width (dense layer for CNN, state size for GRU)')

    # 4. I/O paths
    parser.add_argument('--data-path', type=str, default='data/readout_dataset_2q.pt', help='Path to the readout dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/', help='Directory to save trained model weights')
    parser.add_argument('--results-dir', type=str, default='results/', help='Root folder for figures and sweep results')

    # 5. Physics of the readout (units: microseconds)
    parser.add_argument('--n-qubits', type=int, default=1, choices=[1, 2], help='Number of multiplexed qubits')
    parser.add_argument('--T1', type=float, default=3.0, help='Qubit relaxation time')
    parser.add_argument('--t-ro', type=float, default=2.0, help='Readout window length')
    parser.add_argument('--dt', type=float, default=0.02, help='Sampling interval of the record')
    parser.add_argument('--kappa', type=float, default=10.0, help='Cavity decay rate')
    parser.add_argument('--chi', type=float, default=5.0, help='Dispersive shift')
    parser.add_argument('--eps', type=float, default=5.0, help='Readout drive amplitude')
    parser.add_argument('--sigma', type=float, default=3.0, help='Std of Gaussian noise per sample')
    parser.add_argument('--zeta', type=float, default=1.0, help='Cross-dispersive shift between resonators (2 qubits only)')
    parser.add_argument('--leak', type=float, default=0.1, help='Linear signal leakage between channels (2 qubits only)')

    # 6. Dataset size (records per class; for 2 qubits, per joint state)
    parser.add_argument('--n-train', type=int, default=20000, help='Training records per class')
    parser.add_argument('--n-val', type=int, default=2000, help='Validation records per class')
    parser.add_argument('--n-test', type=int, default=5000, help='Test records per class')

    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    print(parse_args())