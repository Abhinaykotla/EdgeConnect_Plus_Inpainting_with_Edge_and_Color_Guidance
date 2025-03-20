from train_loops import train_g1_and_d1
import multiprocessing

if __name__ == '__main__':
    # Ensure proper multiprocessing behavior on Windows
    multiprocessing.freeze_support()
    
    # Call your training function
    train_g1_and_d1()