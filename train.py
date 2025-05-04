"""
Main training script for EdgeConnect+ inpainting model.

This script serves as the entry point for training either:
- G1: The edge generator model
- G2: The inpainting generator model

The training stage is selectable via command-line arguments, allowing for
sequential training of the two-stage EdgeConnect+ architecture.
"""

import multiprocessing
import argparse

if __name__ == '__main__':
    # Ensure proper multiprocessing behavior on Windows
    # This prevents the script from recursively spawning new processes
    multiprocessing.freeze_support()
    
    # Command line argument parsing for flexible model selection
    parser = argparse.ArgumentParser(description="Train either EdgeConnect+ G1 edge generator or G2 inpainting generator")
    parser.add_argument('--stage', 
                        type=str, 
                        default='g1', 
                        choices=['g1', 'g2'], 
                        help='Which model to train: g1 (edge generator) or g2 (inpainting generator)')
    args = parser.parse_args()
    
    # Import and run the appropriate training function based on user selection
    if args.stage == 'g1':
        # G1 training: Edge map generation from masked inputs
        from train_loops_g1 import train_g1_and_d1
        print("INFO: Starting EdgeConnect G1 (Edge Generator) training...")
        train_g1_and_d1()
    else:  # g2
        # G2 training: Full image inpainting using edge guidance
        from train_loops_g2 import train_g2_and_d2
        print("INFO: Starting EdgeConnect G2 (Inpainting Generator) training...")
        train_g2_and_d2()

    # Note: The commented code below represents alternative import methods
    # that were likely used during development and testing.
    # They are kept for reference but not executed.
    
    # from train_loops_g2 import train_g2_and_d2
    # print("Starting EdgeConnect G2 (Inpainting Generator) training...")
    # train_g2_and_d2()

    # from train_loops_g1 import train_g1_and_d1
    # print("Starting EdgeConnect G1 (Edge Generator) training...")
    # train_g1_and_d1()