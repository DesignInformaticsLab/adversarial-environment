import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

'''
To verify the saved data
'''
if __name__ == '__main__':
    episode_path = 'dynamics/trajectories/random-10000-ns-seed-0-trajectories.npz'
    data = np.load(episode_path)['arr_0']
    states = data['s']
    for i in range(990, 1010):
        plt.imshow(states[i][0], cmap='gray')
        plt.title(f'Frame : {i + 1}')
        plt.show()
