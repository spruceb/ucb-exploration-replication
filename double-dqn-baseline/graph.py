import python_utilities
import json
import csv
from matplotlib import pyplot as plt
import numpy as np
import argparse

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def process_monitor_file(file_path):
    path = python_utilities.check_file(file_path)
    with path.open(newline='') as f:
        reader = csv.reader(f)
        next(reader)
        next(reader)
        data = [line for line in reader]
    data = [(float(length), float(reward)) for (reward, length, _) in data]
    new_data = []
    for length, reward in data:
        if new_data:
            last = new_data[-1][0]
            new_data.append((length + last, reward))
        else:
            new_data.append((length, reward))
    data = new_data
    return data

def graph_average_reward(data, name):
    x, y = zip(*data)
    y = np.array(y)
    K = 100
    y_p = moving_average(y, n=K)
    y[:K - 1] = np.mean(y[:K - 1])
    y[K - 1:] = y_p
    print('Max for {}'.format(name), max(y))
    frames_per_iteration = 4
    plt.plot(np.array(x) * frames_per_iteration, np.array(y), label=name)

def graph_vs_baseline(ucb_data, baseline_data, environment_name):
    plt.clf()
    graph_average_reward(ucb_data, 'UCB')
    graph_average_reward(baseline_data, 'Double DQN')
    plt.legend()
    plt.xlabel('Frames')
    plt.ylabel('100 Episode Mean Reward')
    plt.title('{}'.format(environment_name))
    plt.savefig('{}-comparison.pdf'.format(environment_name))

def get_graph_vs_baseline(parent_path, environment_name,
                       type_names=('baseline', 'multi'), format_string='{}--{}'):
    print(environment_name)
    parent_path = python_utilities.check_directory(parent_path)
    baseline_name, multi_name = type_names
    baseline_name = format_string.format(environment_name, baseline_name)
    multi_name = format_string.format(environment_name, multi_name)
    baseline_path = python_utilities.check_directory(parent_path / baseline_name)
    multi_path = python_utilities.check_directory(parent_path / multi_name)
    baseline_data = process_monitor_file(baseline_path / 'monitor.csv')
    multi_data = process_monitor_file(multi_path / 'monitor.csv')
    graph_vs_baseline(multi_data, baseline_data, environment_name)


def main(args):
    experiments = python_utilities.check_directory(args.experiments_dir)
    environments = args.environments
    for environment in environments:
        get_graph_vs_baseline(experiments, environment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiments-dir', default='finished-experiments')
    parser.add_argument('--environments', nargs='?', default=['SpaceInvaders', 'UpNDown', 'Breakout'])
    args = parser.parse_args()
    main(args)
