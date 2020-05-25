"""
Encapsulate generate data to make it parallel
"""
import numpy as np
import argparse
from multiprocessing import Pool
from subprocess import call

parser = argparse.ArgumentParser('Generate data to train VAE and dynamics model with parallel threads')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--episodes', type=int, default=1000, help='Total number of episodes or rollouts')
parser.add_argument('--threads', type=int, default=5, help='Number of threads')
parser.add_argument('--root-dir', type=str, help='Directory to store data')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--policy', type=str, choices=['random', 'pretrained'], help='Policy to be chosen for agent')
parser.add_argument('--noise-type', type=str, choices=['white', 'brown'], default='brown',
                    help='Noise type used for action sampling')
args = parser.parse_args()

# Generate number of episodes for each thread
q, r = divmod(args.episodes, args.threads)
no_of_episodes = [q + 1] * r + [q] * (args.threads - r)


def _threaded_generation(i):
    thread_seed = np.random.RandomState(i).randint(0, 2 ** 31 - 1)
    cmd = ["python", "dynamics/data/generate.py", "--root-dir", args.root_dir, "--policy", args.policy, "--episodes",
           str(no_of_episodes[i]), "--thread-no", str(i), "--seed", str(thread_seed)]
    if args.render:
        cmd += ["--render"]
    cmd = " ".join(cmd)
    print(cmd)
    call(cmd, shell=True)
    return True


with Pool(processes=args.threads) as pool:
    pool.map(_threaded_generation, range(args.threads))
