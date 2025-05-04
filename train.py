# train.py

import multiprocessing
import argparse

if __name__ == '__main__':
    # Ensure proper multiprocessing behavior on Windows
    multiprocessing.freeze_support()
    
    # Simple command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default='g1', choices=['g1', 'g2'], help='Which model to train: g1 or g2')
    args = parser.parse_args()
    
    # Import the appropriate training module based on the argument
    if args.stage == 'g1':
        from train_loops_g1 import train_g1_and_d1
        print("Starting EdgeConnect G1 (Edge Generator) training...")
        train_g1_and_d1()
    else:  # g2
        from train_loops_g2 import train_g2_and_d2
        print("Starting EdgeConnect G2 (Inpainting Generator) training...")
        train_g2_and_d2()


        # from train_loops_g2 import train_g2_and_d2
        # print("Starting EdgeConnect G2 (Inpainting Generator) training...")
        # train_g2_and_d2()

        # from train_loops_g1 import train_g1_and_d1
        # print("Starting EdgeConnect G1 (Edge Generator) training...")
        # train_g1_and_d1()