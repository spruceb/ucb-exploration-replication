import gym
import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from simple import ActWrapper, learn
import baselines.common.tf_util as U
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common.atari_wrappers import make_atari

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    env = make_atari(args.env)
    # env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
    )
    act = learn(
        env,
        q_func=model,
        lr=1e-4,
        beta1=0.9,
        beta2=0.99,
        epsilon=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=1000000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        print_freq=1,
        batch_size=32,
        learning_starts=50000,
        target_network_update_freq=10000,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
    )
    # act.save("pong_model.pkl") XXX
    env.close()


if __name__ == '__main__':
    main()
