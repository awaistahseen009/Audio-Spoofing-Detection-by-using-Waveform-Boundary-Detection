#!/usr/bin/env python
"""
Training script with command-line arguments
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trainer import main


def parse_args():
    parser = argparse.ArgumentParser(description='Train Audio Spoofing Detection Model')
    parser.add_argument('--extrinsic-dir', type=str, default='new_files/Extrinsic_Partial_Fakes',
                        help='Directory containing extrinsic partial fake audio files')
    parser.add_argument('--intrinsic-dir', type=str, default='new_files/Intrinsic_Partial_Fakes',
                        help='Directory containing intrinsic partial fake audio files')
    parser.add_argument('--real-dir', type=str, default='audio_files/Real/training',
                        help='Directory containing real audio files')
    parser.add_argument('--label-dir', type=str, default='new_files/Generated_Labels',
                        help='Directory containing label files')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--wandb-project', type=str, default='audio-spoofing-detection',
                        help='Weights & Biases project name')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main()
