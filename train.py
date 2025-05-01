# train.py

from train_loops_g2 import train_g2_and_d2
from train_loops_g1 import train_g1_and_d1

import multiprocessing

if __name__ == '__main__':
    # Ensure proper multiprocessing behavior on Windows
    multiprocessing.freeze_support()
    
    # Call your training function
    # train_g1_and_d1()
    train_g2_and_d2()