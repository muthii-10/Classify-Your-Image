#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

def get_input_args():
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    # Training arguments
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16, alexnet)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training')

    # Testing arguments
    parser.add_argument('--input', type=str, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to names')

    return parser.parse_args()
