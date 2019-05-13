# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!

import glob
import json
import os
import argparse
from visdom import Visdom
import subprocess
import re
import time
import csv
from scipy.stats import sem
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})


def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                try:
                    float(tmp[0])
                    float(tmp[2])
                except Exception:
                    continue

                tmp = [float(tmp[2]), int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)
    #y = bin_y(y, bin_size)
    x, y = fix_point(x, y, bin_size)
    return [x, y]


color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


def visdom_plot(viz, win, folder, game, name, num_steps, bin_size=100, smooth=1):
    tx, ty = load_data(folder, smooth, bin_size)
    if tx is None or ty is None:
        return win

    fig = plt.figure()
    plt.plot(tx, ty, label="{}".format(name))

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    plt.title(game)
    plt.legend(loc=4)
    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    return viz.image(image, win=win)


def bin_y(y, bin_size=1):
    return [sum(y[i - min(bin_size, i): i + 1]) / min(bin_size, i + 1) for i in range(len(y))]


def align_data(xs, ys, tick_interval=1000, max_x=1000000):
    min_x = min(min([data_x[-1] for data_x in xs]), max_x)
    new_xs = np.arange(0, int(min_x), tick_interval)
    new_data = np.empty((len(xs), new_xs.shape[0]))
    for i, (xx, yy) in enumerate(zip(xs, ys)):
        new_data[i] = np.interp(new_xs, xx, yy)
    return new_data, new_xs

def organize_data(x, y, name, max_x=None):
    min_x = max_x
    methods = {}
    for xx, yy, n in zip(x, y, name):
        if len(xx) < 1:
            continue
        min_x = min_x if min_x is None or xx[-1] > min_x else xx[-1] 
        if n in methods:
            methods[n]['x'].append(xx)
            methods[n]['y'].append(yy)
        else:
            methods[n] = {}
            methods[n]['x'] = [xx]
            methods[n]['y'] = [yy]
    return methods

def organize_data_x(x, name, max_x=None):
    min_x = max_x
    methods = {}
    for xx,  n in zip(x, name):
        if len(xx) < 1:
            continue
        min_x = min_x if min_x is None or xx[-1] > min_x else xx[-1] 
        if n in methods:
            methods[n]['x'].append(xx)
        else:
            methods[n] = {}
            methods[n]['x'] = [xx]
    return methods

def visdom_plot_batched(viz, win, x, y, title, name, bin_size=1, smooth=1, x_label='Number of Epochs',
                        y_label='Training Error', max_x=None, save_loc=None, ylim=None, log=False, stderr=False):
    if y is None:
        return win
    fig = plt.figure()

    if type(name) is list:
        assert len(x) == len(y) and len(y) == len(name)
        
        y_min = None
        y_max = None
        methods = organize_data(x, y, name, max_x=max_x)
  
        for method in methods:
            m = methods[method]
            data, xs = align_data(m['x'], m['y'], max_x=max_x)
            if stderr:
                error = sem(data, axis=0)
            else:
                error = np.std(data, axis=0)
            mean = np.mean(data, axis=0) 
            if bin_size > 1:
                mean = bin_y(mean, bin_size=bin_size)
            upper = mean + error
            lower = mean - error
            cur_min = np.amin(lower)
            cur_max = np.amax(upper)
       
            y_min = cur_min if y_min is None or cur_min < y_min else y_min
            y_max = cur_max if y_max is None or cur_max > y_max else y_max

            plt.plot(xs, mean, label="{}".format(method))
            if not log:
                plt.fill_between(xs, lower, upper, alpha=0.1)
            else:
                positive = lower > 0
                plt.fill_between(xs, lower, upper, where=positive, alpha=0.1)
    else:
        if bin_size > 1:
            y = bin_y(y, bin_size=bin_size)
        plt.plot(x, y, label="{}".format(name))


    if title.find('NoFrameskip') > -1:
        plt.xticks([1e6, 2e6, 4e6, 6e6, 8e6, 10e6],
                   ["1M", "2M", "4M", "6M", "8M", "10M"], fontsize=14)
        plt.xlim(0, 10e6)
    else:
        plt.xticks([1e5, 2e5, 4e5, 6e5, 8e5, 1e6],
                   ["0.1M", "0.2M", "0.4M", "0.6M", "0.8M", "1M"], fontsize=14)
        plt.xlim(0, 1e6)
    plt.yticks(fontsize=14)

    # if ylim is not None:
    #     plt.ylim(ylim)
    # elif y_min is not None and y_max is not None:
    #     plt.ylim(y_min, y_max)
    
    plt.xlabel(x_label, fontsize=18, fontname='arial')
    plt.ylabel(y_label, fontsize=18, fontname='arial')
    if log:
        plt.yscale('log')
        plt.gca().spines['left']._adjust_location()
            

    plt.legend(loc='upper left', prop={'size' : 12})

    plt.title(title, fontsize=18)
    
    plt.show()
    plt.draw()
    if save_loc is not None and len(x) > 1:
        if '.' in title:
            title = title.replace(".", "-", 1)
        plt.savefig(save_loc + title + '_' + y_label + '.png', bbox_inches='tight')

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    if viz is not None:
        image = np.transpose(image, (2, 0, 1))
        return viz.image(image, win=win)


def load_batched(infile, smooth=1, bin_size=100):
    y = []
    x = []
    
    if '\0' in open(infile).read():
        reader = csv.reader(x.replace('\0', '') for x in infile)
    with open(infile, "r") as f:
        reader = csv.reader(f)
        _, _, max_x = next(reader)
        for row in reader:
            y.append(float(row[-1]))
            x.append(int(round(float(row[0]))))

    return x, y, int(round(float(max_x)))


def add_to_plot_dict(monitor_type, monitor_dict, ylim, game_replay_key, x, y, max_x, algo_name, r_start):
    if monitor_type in monitor_dict[game_replay_key]:
        monitor_dict[game_replay_key][monitor_type]['x'].append(x)
        monitor_dict[game_replay_key][monitor_type]['y'].append(y)
        monitor_dict[game_replay_key][monitor_type]['max_x'].append(max_x)
        monitor_dict[game_replay_key][monitor_type]['algo'].append(algo_name)
        monitor_dict[game_replay_key][monitor_type]['r_start'].append(r_start)
    else:
        monitor_dict[game_replay_key][monitor_type] = {'x': [x], 'y': [y], 'max_x': [max_x],
                                                       'algo': [algo_name],
                                                       'ylim': ylim, 'r_start': [r_start]}


def plot_dict(monitor_dict, viz, save_loc=None, stderr=False):
    vizes = []
    for game_replay_key in monitor_dict:
        for monitor_type in sorted(monitor_dict[game_replay_key]):
            xs = monitor_dict[game_replay_key][monitor_type]['x']
            ys = monitor_dict[game_replay_key][monitor_type]['y']
            max_xs = monitor_dict[game_replay_key][monitor_type]['max_x']
            algos = monitor_dict[game_replay_key][monitor_type]['algo']
            ylim = monitor_dict[game_replay_key][monitor_type]['ylim']
            max_x = max(max_xs)
            x_label = "Number of Steps" if "error" in monitor_type else "Number of Steps"
            log = False
            if "MSE" in monitor_type:
                log = True
                y_label = "TD Error"
            else:
                y_label = "Episode_Reward"
            bin_size = 100 if "MSE" in monitor_type else 1
            vizes.append(visdom_plot_batched(viz, None, xs, ys, title=game_replay_key, name=algos, max_x=max_x,
                                x_label=x_label, y_label=y_label, bin_size=bin_size, save_loc=save_loc, ylim=ylim,
                                             log=log, stderr=stderr))
    return vizes


def get_dirs(log_dirs, dir_key):
    dirs = [d for log_dir in log_dirs for d in glob.glob(log_dir + '*/')]
    dirs.sort()
    dirs = reversed(dirs)
    r_dir = re.compile(dir_key)
    dirs = filter(r_dir.match, dirs)
    dirs = [dir for dir in dirs]
    return dirs


def log_avg_to_csv(monitor_dict, csv_loc, last_hundred=False, random_start=False):
    keys = list(monitor_dict.keys())
    all_algos = [monitor_dict[keys[0]][monitor_type]['algo'] for monitor_type in monitor_dict[keys[0]]][0]

    algos = []
    for algo in all_algos:
        if algo not in algos:
            algos.append(algo)
    print('algos:', algos)
    csv_name = 'avg_results' if not last_hundred else 'asymptotic_results'
    with open(csv_loc + csv_name + '.csv', "wt") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([''] + algos + algos)

    for name in algos:
        csv_name = name + '_per_run_avg_results' if not last_hundred else name + '_per_run_asymptotic_results'
        string_data = ['seed_' + str(i) for i in range(10)]
        with open(csv_loc + csv_name + '.csv', "wt") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([''] + string_data)

    if random_start:
        for name in algos:
            csv_name = name + '_per_run_random_results'
            string_data = ['seed_' + str(i) for i in range(10)]
            with open(csv_loc + csv_name + '.csv', "wt") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([''] + string_data)
    
    for game_replay_key in sorted(monitor_dict):
        for monitor_type in sorted(monitor_dict[game_replay_key]):
            xs = monitor_dict[game_replay_key][monitor_type]['x']
            ys = monitor_dict[game_replay_key][monitor_type]['y']
            names = monitor_dict[game_replay_key][monitor_type]['algo']
            r_starts = monitor_dict[game_replay_key][monitor_type]['r_start']
            methods = organize_data(xs, ys, names)
            methods_r_starts = organize_data_x(r_starts, names)
            avg_data = np.empty((len(methods)))
            std_data = np.empty((len(methods)))
            per_seed_data = np.empty((len(methods), int(len(xs)/ len(methods))))
            per_seed_data_random = np.empty((len(methods), int(len(xs)/ len(methods))))
            for i, m in enumerate(methods):
                x = methods[m]['x']
                y = methods[m]['y']
                r_start = methods_r_starts[m]['x']

                cur_data = np.empty((len(y)))
                if last_hundred:
                    for j in range(len(y)):
                        cur_mean = np.mean(y[j][-100:])
                        cur_data[j] = cur_mean
                        per_seed_data[i, j] = cur_mean
                else:
                    for j in range(len(y)):
                        cur_mean = np.mean(y[j])
                        cur_data[j] = cur_mean
                        per_seed_data[i, j] = cur_mean

                avg_data[i] = np.mean(cur_data)
                std_data[i] = np.std(cur_data)

                if random_start:
                    for j in range(len(r_start)):
                        per_seed_data_random[i, j] = r_start[j][0]
                    

            # data = reshaped_data(data, monitor_dict[game_replay_key][monitor_type]['algo'], algos)
            mean = [str(avg) for avg in avg_data]
            std = [str(s) for s in std_data]
            csv_name = 'avg_results' if not last_hundred else 'asymptotic_results'
            with open(csv_loc + csv_name + '.csv', "a") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([game_replay_key] + mean + std)

            for per_seed_data_method, name in zip(per_seed_data, methods):
                csv_name = name + '_per_run_avg_results' if not last_hundred else name + '_per_run_asymptotic_results'
                string_data = [str(d) for d in per_seed_data_method]
                with open(csv_loc + csv_name + '.csv', "a") as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([game_replay_key] + string_data) 

            for per_seed_data_method_random, name in zip(per_seed_data_random, methods):
                csv_name = name + '_per_run_random_results'
                string_data = [str(d) for d in per_seed_data_method_random]
                with open(csv_loc + csv_name + '.csv', "a") as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([game_replay_key] + string_data) 



# Format for log directory should be should be log_dir + game_name + "_" + algo_name + run + "/"
def visdom_plot_all(viz, log_dir, dir_key='.*', monitor_key='.*', plot_other=False, mode_name=None, mode_list=None,
                    save_loc=None, log_avg=False, log_variance=False, plot=True, last_hundred=False, stderr=False,
                    random_start=False):
    monitor_dict = {}
    game_ylims = {}
    dir_ext = '' 
    for d, dir in enumerate(get_dirs(log_dir, dir_key + dir_ext + '/')):
        local_dir = dir[:-1]
        local_dir = local_dir[local_dir.rfind('/') + 1:]
        game_name = local_dir[:local_dir.find('_')]
        algo_name = local_dir[local_dir.rfind('_') + 1:]
        
        algo_name = algo_name.replace('Baseline', '')
        algo_name = algo_name.replace('Gae', '')
        algo_name = algo_name.replace('Value', '')
        algo_name = algo_name.replace('Gamma', '')
        algo_name = algo_name.replace('CappedBias', 'CappedLambda')


        game_replay_key = game_name 

        if game_replay_key not in monitor_dict:
            monitor_dict[game_replay_key] = {}

        if random_start:
            with open(dir + 'random_rewards.pkl', 'rb') as handle:
                r_start = pickle.load(handle)

            r_start = [sum(r_start) / len(r_start)]
        else:
            r_start =[None]

        print(dir)
        tx, ty = load_data(dir, smooth=1, bin_size=100)

        if tx is not None and ty is not None:
            if game_name not in game_ylims:
                game_ylims[game_name] = [int(min(ty)), int(max(ty))]
            else:
                game_ylims[game_name][0] = np.min([int(np.min(ty)), game_ylims[game_name][0]])
                game_ylims[game_name][1] = np.max([int(np.max(ty)), game_ylims[game_name][1]])

            add_to_plot_dict('Reward', monitor_dict, game_ylims[game_name], game_replay_key, tx, ty, int(1e7), algo_name, r_start)


    if log_variance:
        log_variance_to_csv(monitor_dict, save_loc)

    if log_avg:
        log_avg_to_csv(monitor_dict, csv_loc=save_loc, last_hundred=last_hundred, random_start=random_start)

    if plot:
        return plot_dict(monitor_dict, viz, save_loc=save_loc, stderr=stderr)


def parse_args():
    parser = argparse.ArgumentParser("visdom parser")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--log-dir", type=str, nargs='*', default=['./logs/'],
                        help="directory in which training state and model should be saved.")
    parser.add_argument("--vis-interval", type=int, default=None,
                        help="num seconds between vis plotting, default just one plot")
    parser.add_argument("--log-avg", action='store_true', default=False)
    parser.add_argument("--last-hundred", action='store_true', default=False)
    parser.add_argument("--plot-other", action='store_true', default=False)
    parser.add_argument("--dir-key", type=str,  default='.*',
                        help="directory filter")
    parser.add_argument("--stderr", action='store_true',  default=False,
                        help="plot standard error")
    parser.add_argument("--random-start", action='store_true',  default=False,
                        help="gen random start data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot = True if not args.log_avg else False
    viz = Visdom(port=args.port)
    # if not viz.check_connection():
    #     subprocess.Popen(["python", "-m", "visdom.server", "-p", str(args.port), "-logging_level", "ERROR"])

    if args.vis_interval is not None:
        vizes = []
        while True:
            for v in vizes:
                viz.close(v)
            try:
                vizes = visdom_plot_all(viz, args.log_dir, mode_name='', mode_list=[],
                                        save_loc=None, plot_other=False, log_avg=args.log_avg, plot=plot, last_hundred=args.last_hundred,
                                        dir_key=args.dir_key, stderr=args.stderr, random_start=args.random_start)
            except IOError:
                pass
            time.sleep(args.vis_interval)
    else:
        visdom_plot_all(viz, args.log_dir, mode_name='', mode_list=[],
                        save_loc='./plots/', plot_other=args.plot_other, log_avg=args.log_avg, plot=plot, last_hundred=args.last_hundred, dir_key=args.dir_key, stderr=args.stderr, random_start=args.random_start)
