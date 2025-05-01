# train.py

from train_loops_g2 import train_g2_and_d2
from train_loops_g1 import train_g1_and_d1

import multiprocessing

if __name__ == '__main__':
    # Ensure proper multiprocessing behavior on Windows
    multiprocessing.freeze_support()
    
    # Call your training function
    train_g1_and_d1()
    # train_g2_and_d2()

# import argparse
# import multiprocessing

# if __name__ == '__main__':
#     multiprocessing.freeze_support()

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--stage', type=str, default='g1', choices=['g1', 'g2'], help='Which model to train: g1 or g2')
#     args = parser.parse_args()

#     if args.stage == 'g1':
#         from train_loops_g1 import train_g1_and_d1
#         train_g1_and_d1()
#     elif args.stage == 'g2':
#         from train_loops_g2 import train_g2_and_d2
#         train_g2_and_d2()
#     else:
#         raise ValueError("Invalid stage. Choose 'g1' or 'g2'.")