# parser.py

import os
import argparse


def get_process_args():
    # Function to get command-line arguments

    # initialize argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--input-dir',
                        type=str,
                        help='Path to training data.')
    parser.add_argument('--output-dir',
                        type=str,
                        default=None,
                        help='Directory to save processed slogs.')
    parser.add_argument('--freq',
                        type=str,
                        default='1min',
                        help='Frequency to resample events together.')
    parser.add_argument('--tau',
                        type=float,
                        default=0.5,
                        help='Percentage to marge tokens')
    # parse args
    args = parser.parse_args()

    # create saved folder if not existing
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args

def get_model_args():
    # Function to get command-line arguments

    # initialize argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--data',
                        type=str,
                        help='Path to training data.')
    parser.add_argument('--eval-data',
                        type=str,
                        default=None,
                        help='Path to evaluation data.')
    parser.add_argument('--shuffle',
                        default=False,
                        action='store_true',
                        help='Flag to shuffle dataset.')
    parser.add_argument('--iter',
                        type=int,
                        default=50,
                        help='Number of iterations')
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='Number of samples per batch.')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.01,
                        help='Learning rate.')
    parser.add_argument('--lr-decay',
                        default=False,
                        action='store_true',
                        help='Flag to use the learning-rate decay.')
    parser.add_argument('--saved',
                        type=str,
                        default='./saved',
                        help='Directory to save trained models, params, and logs.')

    # parse args
    args = parser.parse_args()

    # create saved folder if not existing
    if not os.path.exists(args.saved):
        os.makedirs(args.saved)

    return args
