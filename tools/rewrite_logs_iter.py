# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/open-
mmlab/mmdetection/blob/master/tools/analysis_tools/analyze_logs.py."""
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns


def rewrite(log_dicts, args, step):
    for i, log_dict in enumerate(log_dicts):
        new_log = log_dict
        # print(log_dict)                     
        iters = list(log_dict.keys())
        epoch = 1
        total_step = step
        for iter in iters:
            # new_log[]
            # print(log_dict.keys())
            # iter_logs = log_dict[iter]
            # print('iter log',  iter_logs)
            new_log[iter]['iter'] = iter
            new_log[iter]['epoch'] = epoch

            if iter >= total_step:
                total_step += step
                epoch += 1
            # print('epoch', iter_logs['epoch'][0]) #epoch 4
            # print('epoch', iter_logs['epoch']) #epoch [4]
            with open(args.out, 'a+') as f:
                # new_log = dict(new_log)
                # print(new_log.)
                json.dump(new_log[iter], f)

    # print('out', args.out)

def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            plot_epochs = []
            plot_iters = []
            plot_values = []
            # In some log files, iters number is not correct, `pre_iter` is
            # used to prevent generate wrong lines.
            pre_iter = -1
            for epoch in epochs:
                epoch_logs = log_dict[epoch]
                if metric not in epoch_logs.keys():
                    continue
                if metric in ['a1', 'a2', 'a3', 'abs_rel', 'rmse', 'log_10', 'rmse_log', 'silog', 'sq_rel']:
                    plot_epochs.append(epoch)
                    plot_values.append(epoch_logs[metric][0])
                else:
                    for idx in range(len(epoch_logs[metric])):
                        if pre_iter > epoch_logs['iter'][idx]:
                            continue
                        pre_iter = epoch_logs['iter'][idx]
                        plot_iters.append(epoch_logs['iter'][idx])
                        plot_values.append(epoch_logs[metric][idx])
            ax = plt.gca()
            label = legend[i * num_metrics + j]
            if metric in ['a1', 'a2', 'a3', 'abs_rel', 'rmse', 'log_10', 'rmse_log', 'silog', 'sq_rel']:
                ax.set_xticks(plot_epochs)
                plt.xlabel('epoch')
                plt.plot(plot_epochs, plot_values, label=label, marker='o')
            else:
                plt.xlabel('iter')
                plt.plot(plot_iters, plot_values, label=label, linewidth=0.5)
        plt.legend()
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    parser.add_argument(
        '--json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser.add_argument('--title', type=str, help='title of figure')
    parser.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser.add_argument('--out', type=str, default=None)  #save fig
    args = parser.parse_args()
    return args

# if metric in ['a1', 'a2', 'a3', 'abs_rel', 'rmse', 'log_10', 'rmse_log', 'silog', 'sq_rel']:
def load_json_logs(json_logs, args, step):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics
    # value of sub dict is a list of corresponding values of all iterations
    # log_dicts = [dict() for _ in json_logs]
    epoch = 1
    total_step = 0
    best_rel = 100
    best_rel_epoch = 0

    best_all = 100
    best_epoch = 0
    best_epoch_abs = 0

    for json_log in json_logs:
        with open(json_log, 'r') as log_file:
            # new_log = log_file
            for index, line in enumerate(log_file):
                log = json.loads(line.strip())
                # print(log)

                if log['mode'] == 'val':
                    if log['abs_rel'] < best_rel:
                        best_rel = log['abs_rel']
                        best_rel_epoch = log['epoch']

                    all = 1-log['a1']+log['abs_rel']+log['rmse']
                    # all = 3-log['a1']-log['a2']-log['a3']+log['abs_rel']+log['rmse']+log['log_10']+log['rmse_log']+log['sq_rel']
                    if all < best_all:
                        best_all = all
                        best_epoch = log['epoch']
                        best_epoch_abs = log['abs_rel']

                if log['mode'] == 'train':
                    total_step += step

                if log['mode'] == 'train' and log['iter'] < total_step:
                    log['iter'] = total_step

                
                with open(args.out, 'a+') as f:
                    # new_log = dict(new_log)
                    # print(new_log.)
                    json.dump(log, f)
                    f.write('\n')
    print('best all', best_all, 'epoch', best_epoch, 'abs', best_epoch_abs)
    print('best abs', best_rel, 'epoch', best_rel_epoch)

# def load_json_logs(json_logs):
#     # load and convert json_logs to log_dict, key is epoch, value is a sub dict
#     # keys of sub dict is different metrics
#     # value of sub dict is a list of corresponding values of all iterations
#     log_dicts = [dict() for _ in json_logs]
#     for json_log, log_dict in zip(json_logs, log_dicts):
#         with open(json_log, 'r') as log_file:
#             new_log = log_file
#             for line in log_file:
#                 log = json.loads(line.strip())
#                 # skip lines without `epoch` field
#                 if 'epoch' not in log:
#                     continue
#                 epoch = log.pop('iter')
#                 if epoch not in log_dict:
#                     log_dict[epoch] = defaultdict(list)
#                 for k, v in log.items():
#                     log_dict[epoch][k].append(v)
#     return log_dicts


def main():
    # step = 6050 #96 4 batch
    step = 50 #24 6 batch
    # step = 4000 #64 6 batch
    args = parse_args()
    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')
    load_json_logs(json_logs, args, step)
    # log_dicts = load_json_logs(json_logs)
    # plot_curve(log_dicts, args)
    # rewrite(log_dicts, args, step)


if __name__ == '__main__':
    main()
