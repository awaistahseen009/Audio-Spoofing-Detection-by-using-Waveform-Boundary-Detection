#!/usr/bin/env python
"""
Inference script with command-line arguments
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import main_inference


def parse_args():
    parser = argparse.ArgumentParser(description='Run Audio Spoofing Detection Inference')
    parser.add_argument('--audio-path', type=str, required=True,
                        help='Path to audio file for inference')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/checkpoint_epoch_4.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    result = main_inference(args.audio_path, args.checkpoint_path)
    
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Audio File: {args.audio_path}")
    print(f"Total Frames: {result['total_frames']}")
    print(f"Fake Frames: {result['fake_frames']}")
    print(f"Fake Percentage: {result['fake_percentage']:.2f}%")
    print(f"Classification: {'SPOOFED' if result['fake_percentage'] > 50 else 'GENUINE'}")
    print("="*50)
