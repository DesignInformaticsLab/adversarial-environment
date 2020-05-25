import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

'''
To verify the saved data
'''
if __name__ == '__main__':
    episode_path = 'datasets/carracing/random/brown/thread_0/episode_2.npz'
    data = np.load(episode_path)
    states = data['states']
    for i in range(900, 1000):
        plt.imshow(states[i][0], cmap='gray')
        plt.title(f'Frame : {i + 1}')
        plt.show()
