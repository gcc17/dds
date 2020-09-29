import numpy as np
import os
import matplotlib.pyplot as plt
from dds_utils import (read_results_dict, Results)
from .plot_utils import (Ellipse, get_colors, read_stats_costs, confidence_ellipse, pick_list_item, \
    sort_by_second_list, find_best_stat_point, get_markers, each_trick_point, simple_cull, dominates_stat_acc, \
    get_all_videos_data)
import pandas as pd
import shutil
import ipdb


def compare_all_acc_cost(stats_path, costs_path, stats_metric='F1', costs_metric='frame-count',
        video_names=['trafficcam_1'], save_direc=None, save_fname=None, 
        compare_mpeg=False, mpeg_key='mpeg_0.8_26', 
        context_type_idx=0, context_val_idx=1, blank_type_idx=2, blank_val_idx=3, 
        area_upper_bound_idx=5, resize_type_idx=6, resize_val_idx=7, inter_iou_idx=9):
    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'
    stats_list, costs_list = read_stats_costs(stats_path, costs_path)
    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=video_names)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=video_names)
    print(len(stats_result_dict), len(costs_baseline_result_dict))

    dds_cost_dict = costs_baseline_result_dict['dds']
    dds_stat_dict = stats_baseline_result_dict['dds']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # dds points
    # use the dds point for normalization
    for cost_key, cost_val_dict in costs_baseline_result_dict.items():
        baseline_stats_data = []
        baseline_costs_data = []
        for cur_video_name in video_names:
            if cur_video_name in cost_val_dict.keys():
                cur_dds_cost = dds_cost_dict[cur_video_name]
                cur_dds_stat = dds_stat_dict[cur_video_name]
                cost_val = cost_val_dict[cur_video_name] / cur_dds_cost
                stat_val = stats_baseline_result_dict[cost_key][cur_video_name] / cur_dds_stat
                baseline_costs_data.append(cost_val)
                baseline_stats_data.append(stat_val)
        ax.scatter(baseline_costs_data, baseline_stats_data, label=cost_key)
        # print(baseline_costs_data, baseline_stats_data)       

    # best points
    for cur_video_name in video_names:
        best_cost, best_stat, best_method_name = \
            find_best_stat_point(stats_result_dict, costs_result_dict, cur_video_name, choose_metric)
        cur_dds_cost = dds_cost_dict[cur_video_name]
        cur_dds_stat = dds_stat_dict[cur_video_name]
        ax.scatter(best_cost/cur_dds_cost, best_stat/cur_dds_stat, label=best_method_name, marker='x')
    
    # no padding and no resizing
    ratio1_area_bound = None
    no_trick_stats_data = []
    no_trick_costs_data = []
    no_trick_keys = []
    for method_name, method_result_dict in costs_result_dict.items():
        cur_para_list = (str(method_name)).split('_')
        context_val = float(cur_para_list[context_val_idx])
        blank_val = float(cur_para_list[blank_val_idx])
        if not (context_val == 0.0 and blank_val == 0.0):
            continue
        resize_type = cur_para_list[resize_type_idx]
        resize_val = float(cur_para_list[resize_val_idx])
        cur_area_upper_bound = float(cur_para_list[area_upper_bound_idx])
        if not (resize_type == 'ratio' and resize_val == 1):
            continue
        if ratio1_area_bound is None:
            ratio1_area_bound = cur_area_upper_bound
        if ratio1_area_bound != cur_area_upper_bound:
            continue
        
        for cur_video_name in video_names:
            if cur_video_name in method_result_dict.keys():
                cur_dds_cost = dds_cost_dict[cur_video_name]
                cur_dds_stat = dds_stat_dict[cur_video_name]
                cost_val = method_result_dict[cur_video_name] / cur_dds_cost
                stat_val = stats_result_dict[method_name][cur_video_name] / cur_dds_stat
                if compare_mpeg:
                    mpeg_stat_val = stats_baseline_result_dict[mpeg_key][cur_video_name] / cur_dds_stat
                    if not greater_than_target(mpeg_stat_val, stat_val, choose_metric):
                        continue
                no_trick_costs_data.append(cost_val)
                no_trick_stats_data.append(stat_val)
                no_trick_keys.append(method_name)
    ax.scatter(no_trick_costs_data, no_trick_stats_data, label='no_trick', c='purple')
    if no_trick_costs_data and no_trick_stats_data:
        confidence_ellipse(np.array(no_trick_costs_data), np.array(no_trick_stats_data), \
            ax, n_std=1, label='no_trick', edgecolor='purple')
    
    print(no_trick_stats_data)
    # print(no_trick_costs_data)
    print(no_trick_keys)

    # only padding and no resizing
    only_padding_stats_data = []
    only_padding_costs_data = []
    only_padding_keys = []
    for method_name, method_result_dict in costs_result_dict.items():
        cur_para_list = (str(method_name)).split('_')
        resize_type = cur_para_list[resize_type_idx]
        resize_val = float(cur_para_list[resize_val_idx])
        cur_area_upper_bound = float(cur_para_list[area_upper_bound_idx])
        if not (resize_type == 'ratio' and resize_val == 1):
            continue
        if ratio1_area_bound is None:
            ratio1_area_bound = cur_area_upper_bound
        if ratio1_area_bound != cur_area_upper_bound:
            continue

        for cur_video_name in video_names:
            if cur_video_name in method_result_dict.keys():
                cur_dds_cost = dds_cost_dict[cur_video_name]
                cur_dds_stat = dds_stat_dict[cur_video_name]
                cost_val = method_result_dict[cur_video_name] / cur_dds_cost
                stat_val = stats_result_dict[method_name][cur_video_name] / cur_dds_stat
                if compare_mpeg:
                    mpeg_stat_val = stats_baseline_result_dict[mpeg_key][cur_video_name] / cur_dds_stat
                    if not greater_than_target(mpeg_stat_val, stat_val, choose_metric):
                        continue
                only_padding_costs_data.append(cost_val)
                only_padding_stats_data.append(stat_val)
                only_padding_keys.append(method_name)
    ax.scatter(only_padding_costs_data, only_padding_stats_data, s=0.5, label='only_padding', c='blue')
    if only_padding_costs_data and only_padding_stats_data:
        confidence_ellipse(np.array(only_padding_costs_data), np.array(only_padding_stats_data), \
            ax, n_std=1, label='only_padding', edgecolor='blue', linestyle=':')
    # print(only_padding_stats_data)
    # print(only_padding_costs_data)
    # print(only_padding_keys)
    
    # only resizing and no padding
    only_resizing_stats_data = []
    only_resizing_costs_data = []
    only_resizing_keys = []
    for method_name, method_result_dict in costs_result_dict.items():
        cur_para_list = (str(method_name)).split('_')
        context_val = float(cur_para_list[context_val_idx])
        blank_val = float(cur_para_list[blank_val_idx])
        if not (context_val == 0.0 and blank_val == 0.0):
            continue
        resize_type = cur_para_list[resize_type_idx]
        resize_val = float(cur_para_list[resize_val_idx])
        if resize_type == 'ratio' and resize_val == 1:
            continue

        for cur_video_name in video_names:
            if cur_video_name in method_result_dict.keys():
                cur_dds_cost = dds_cost_dict[cur_video_name]
                cur_dds_stat = dds_stat_dict[cur_video_name]
                cost_val = method_result_dict[cur_video_name] / cur_dds_cost
                stat_val = stats_result_dict[method_name][cur_video_name] / cur_dds_stat
                if compare_mpeg:
                    mpeg_stat_val = stats_baseline_result_dict[mpeg_key][cur_video_name] / cur_dds_stat
                    if not greater_than_target(mpeg_stat_val, stat_val, choose_metric):
                        continue
                only_resizing_costs_data.append(cost_val)
                only_resizing_stats_data.append(stat_val)
                only_resizing_keys.append(method_name)
    ax.scatter(only_resizing_costs_data, only_resizing_stats_data, s=0.5, label='only_resizing', c='fuchsia')
    if only_resizing_costs_data and only_resizing_stats_data:
        confidence_ellipse(np.array(only_resizing_costs_data), np.array(only_resizing_stats_data), \
            ax, n_std=1, label='only_resizing', edgecolor='fuchsia', linestyle='--')
    # print(only_resizing_stats_data)
    # print(only_resizing_costs_data)
    # print(only_resizing_keys)   

    # padding and resizing
    padding_resizing_stats_data = []
    padding_resizing_costs_data = []
    padding_resizing_keys = []
    for method_name, method_result_dict in costs_result_dict.items():
        cur_para_list = (str(method_name)).split('_')
        context_val = float(cur_para_list[context_val_idx])
        blank_val = float(cur_para_list[blank_val_idx])
        if (context_val == 0.0 and blank_val == 0.0):
            continue
        resize_type = cur_para_list[resize_type_idx]
        resize_val = float(cur_para_list[resize_val_idx])
        if resize_type == 'ratio' and resize_val == 1:
            continue
        
        for cur_video_name in video_names:
            if cur_video_name in method_result_dict.keys():
                cur_dds_cost = dds_cost_dict[cur_video_name]
                cur_dds_stat = dds_stat_dict[cur_video_name]
                cost_val = method_result_dict[cur_video_name] / cur_dds_cost
                stat_val = stats_result_dict[method_name][cur_video_name] / cur_dds_stat
                if compare_mpeg:
                    mpeg_stat_val = stats_baseline_result_dict[mpeg_key][cur_video_name] / cur_dds_stat
                    if not greater_than_target(mpeg_stat_val, stat_val, choose_metric):
                        continue
                padding_resizing_costs_data.append(cost_val)
                padding_resizing_stats_data.append(stat_val)
                padding_resizing_keys.append(method_name)

    ax.scatter(padding_resizing_costs_data, padding_resizing_stats_data, s=0.5, label='padding_resizing', c='firebrick')
    if padding_resizing_costs_data and padding_resizing_stats_data:
        confidence_ellipse(np.array(padding_resizing_costs_data), np.array(padding_resizing_stats_data), \
            ax, n_std=1, label='padding_resizing', edgecolor='firebrick')
    # print(padding_resizing_stats_data[:100])
    # print(padding_resizing_costs_data[:100])
    # print(padding_resizing_keys)

    videos = '_'.join(video_names)
    plt.title(f'{videos}_{stats_metric}_{costs_metric}')
    ax.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0)
    ax.set_xlabel(f'relative_{costs_metric}')
    ax.set_ylabel(f'relative_{stats_metric}')
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def compare_padding_method(stats_path, costs_path, stats_metric='F1', costs_metric='frame-count', 
        video_names=['trafficcam_1'], save_direc=None, save_fname=None, 
        context_type_idx=0, blank_type_idx=2):
    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'
    stats_list, costs_list = read_stats_costs(stats_path, costs_path)
    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=video_names)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=video_names)
    print(len(stats_result_dict), len(costs_baseline_result_dict))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # baseline methods
    for cost_key, cost_val_dict in costs_baseline_result_dict.items():
        baseline_stats_data = []
        baseline_costs_data = []
        for cur_video_name in video_names:
            if cur_video_name in cost_val_dict.keys():
                cost_val = cost_val_dict[cur_video_name]
                stat_val = stats_baseline_result_dict[cost_key][cur_video_name]
                baseline_costs_data.append(cost_val)
                baseline_stats_data.append(stat_val)
        ax.scatter(baseline_costs_data, baseline_stats_data, label=cost_key)

    padding_methods_stats_dict = {}
    padding_methods_costs_dict = {}
    for method_name, method_result_dict in costs_result_dict.items():
        cur_para_list = (str(method_name)).split('_')
        context_type = cur_para_list[context_type_idx]
        blank_type = cur_para_list[blank_type_idx]
        padding_method_key = f'{context_type}-{blank_type}'
        if padding_method_key not in padding_methods_costs_dict.keys():
            padding_methods_stats_dict[padding_method_key] = []
            padding_methods_costs_dict[padding_method_key] = []
        
        for cur_video_name in video_names:
            if cur_video_name in method_result_dict.keys():
                cost_val = method_result_dict[cur_video_name]
                stat_val = stats_result_dict[method_name][cur_video_name]
                padding_methods_costs_dict[padding_method_key].append(cost_val)
                padding_methods_stats_dict[padding_method_key].append(stat_val)
    
    padding_methods_stats_dict = dict(sorted(padding_methods_stats_dict.items(),key=lambda x:x[0]))
    padding_methods_costs_dict = dict(sorted(padding_methods_costs_dict.items(),key=lambda x:x[0]))
    color_list = get_colors(len(padding_methods_costs_dict))
    idx = 0
    for padding_method_key, padding_method_cost_list in padding_methods_costs_dict.items():
        padding_method_stat_list = padding_methods_stats_dict[padding_method_key]
        padding_method_stat_list, padding_method_cost_list = sort_by_second_list(
            padding_method_stat_list, padding_method_cost_list
        )
        ax.plot(padding_method_cost_list, padding_method_stat_list, label=padding_method_key, \
            c=color_list[idx])
        idx += 1

        cur_stat_dict = {padding_method_key: padding_method_stat_list}
        cur_cost_dict = {padding_method_key: padding_method_cost_list}
        # Each padding method best points
    
    videos = '_'.join(video_names)
    plt.title(f'{videos}_{stats_metric}_{costs_metric}')
    ax.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0)
    ax.set_xlabel(f'absolute_{costs_metric}')
    ax.set_ylabel(f'absolute_{stats_metric}')
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def each_padding_compare(stats_path, costs_path, stats_metric='F1', costs_metric='frame-count', 
        target_video_name='trafficcam_1', save_direc=None, save_fname=None, 
        context_type='inverse', blank_type='inverse',
        context_type_idx=0, context_val_idx=1, blank_type_idx=2, blank_val_idx=3, 
        area_upper_bound_idx=5, resize_type_idx=6, resize_val_idx=7, inter_iou_idx=9):
    
    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'
    stats_list, costs_list = read_stats_costs(stats_path, costs_path)
    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=target_video_name, context_type=context_type, blank_type=blank_type)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_name, context_type=context_type, blank_type=blank_type)
    print(len(stats_result_dict), len(costs_baseline_result_dict))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # use color to represent context padding value 
    # use marker to represent blank padding value
    context_val_list = []
    blank_val_list = []
    for method_name in stats_result_dict.keys():
        cur_para_list = (str(method_name)).split('_')
        context_val = float(cur_para_list[context_val_idx])
        blank_val = float(cur_para_list[blank_val_idx])
        if blank_val == 0.001 and blank_type == 'inverse' and context_type == 'inverse':
            continue
        if context_val not in context_val_list:
            context_val_list.append(context_val)
        if blank_val not in blank_val_list:
            blank_val_list.append(blank_val)
        
    context_val_list.sort()
    blank_val_list.sort()
    
    color_list = get_colors(len(context_val_list))
    marker_list = get_markers(len(blank_val_list))
    if target_video_name in costs_baseline_result_dict['dds'].keys():
        set_x = costs_baseline_result_dict['dds'][target_video_name]
        set_y = stats_baseline_result_dict['dds'][target_video_name]
    else:
        set_x = 1
        set_y = 300
    for idx, context_val in enumerate(context_val_list):
        ax.scatter(set_x, set_y, c=color_list[idx], label=f'context-{context_val}')
    for idx, blank_val in enumerate(blank_val_list):
        ax.scatter(set_x, set_y, c='black', marker=marker_list[idx], label=f'blank-{blank_val}')
    
    # baseline methods
    for cost_key, cost_val_dict in costs_baseline_result_dict.items():
        if target_video_name in cost_val_dict.keys():
            cost_val = cost_val_dict[target_video_name]
            stat_val = stats_baseline_result_dict[cost_key][target_video_name]
            ax.scatter(cost_val, stat_val, label=cost_key)
    
    for method_name, method_result_dict in costs_result_dict.items():
        cur_para_list = (str(method_name)).split('_')
        context_val = float(cur_para_list[context_val_idx])
        blank_val = float(cur_para_list[blank_val_idx])
        resize_type = cur_para_list[resize_type_idx]
        resize_ratio = float(cur_para_list[resize_val_idx])
        if blank_val == 0.001 and blank_type == 'inverse' and context_type == 'inverse':
            continue
        if not (resize_type == 'ratio' and resize_ratio == 1):
            continue
        for context_idx in range(len(context_val_list)):
            if context_val_list[context_idx] == context_val:
                break
        for blank_idx in range(len(blank_val_list)):
            if blank_val_list[blank_idx] == blank_val:
                break
        
        if target_video_name in method_result_dict.keys():
            cost_data = method_result_dict[target_video_name]
            stat_data = stats_result_dict[method_name][target_video_name]
            ax.scatter(cost_data, stat_data, c=color_list[context_idx], marker=marker_list[blank_idx])
    
    plt.title(f'{target_video_name}_{context_type}_{blank_type}')
    ax.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0)
    ax.set_xlabel(f'absolute_{costs_metric}')
    ax.set_ylabel(f'absolute_{stats_metric}')
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_trick_point(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count', \
        new_stat=False, dds_as_gt=False, save_direc=None, save_fname=None):
    
    stats_mean_baseline_dict, costs_mean_baseline_dict, \
        no_trick_stat, no_trick_cost, no_trick_method, \
        padding_stat, padding_cost, padding_method, resizing_stat, resizing_cost, resizing_method, \
        merge_stat, merge_cost, merge_method,\
        pad_resize_stat, pad_resize_cost, pad_resize_method, pad_merge_stat, pad_merge_cost, pad_merge_method,\
        resize_merge_stat, resize_merge_cost, resize_merge_method, \
        all_tricks_stat, all_tricks_cost, all_tricks_method = each_trick_point(
            file_direc, target_video_list, stats_metric, costs_metric, new_stat, dds_as_gt
        )

    fig = plt.figure()
    ax = fig.add_subplot(111)  

    for method_name, baseline_mean_cost in costs_mean_baseline_dict.items():
        baseline_mean_stat = stats_mean_baseline_dict[method_name]
        if dds_as_gt and method_name == 'dds' and new_stat:
            continue
        ax.scatter(baseline_mean_cost, baseline_mean_stat, label=method_name)
        ax.text(baseline_mean_cost, baseline_mean_stat, method_name)

    ax.scatter(no_trick_cost, no_trick_stat, label='no_trick')
    ax.text(no_trick_cost, no_trick_stat, 'no_trick')
    ax.scatter(padding_cost, padding_stat, label='pad')
    ax.text(padding_cost, padding_stat*0.998, 'pad')
    ax.scatter(resizing_cost, resizing_stat, label='resize')
    ax.text(resizing_cost, resizing_stat, 'resize')
    ax.scatter(merge_cost, merge_stat, label='merge')
    ax.text(merge_cost, merge_stat, 'merge')
    ax.scatter(pad_resize_cost, pad_resize_stat, label='pad-resize')
    ax.text(pad_resize_cost, pad_resize_stat, 'pad-resize')
    ax.scatter(pad_merge_cost, pad_merge_stat, label='pad-merge')
    ax.text(pad_merge_cost, pad_merge_stat*0.998, 'pad-merge')
    ax.scatter(resize_merge_cost, resize_merge_stat, label='resize-merge')
    ax.text(resize_merge_cost, resize_merge_stat, 'resize-merge')
    ax.scatter(all_tricks_cost, all_tricks_stat, label='all-trick')
    ax.text(all_tricks_cost, all_tricks_stat*1.002, 'all-trick')

    if new_stat:
        name_str = '2nd'
    else:
        name_str = 'overall'
    if dds_as_gt:
        name_str = name_str + '-dds'
    else:
        name_str = name_str + '-gt'

    plt.title(name_str)
    ax.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0)
    ax.set_xlabel(f'average_{costs_metric}')
    ax.set_ylabel(f'average_{stats_metric}')
    if save_direc:
        os.makedirs(save_direc, exist_ok=True)
        if not save_fname:
            save_fname = f'{name_str}.png'
        plt.savefig(os.path.join(save_direc, save_fname), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_pareto_line(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count', 
        dds_as_gt=False, save_direc=None, save_fname=None, compare_cost=True):

    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.set_ylabel(f'overall-{stats_metric}')
    ax2 = ax1.twinx()
    ax2.set_ylabel(f'2nd-{stats_metric}')
    line_colors = get_colors(2)
    if dds_as_gt:
        ax2.set_title(f'{stats_metric}-overall_gt-2nd_dds')
    else:
        ax2.set_title(f'{stats_metric}-overall_gt-2nd_gt')
    ori_dds_as_gt = dds_as_gt

    for new_stat in [False, True]:
        if not new_stat:
            dds_as_gt = False
        else:
            dds_as_gt = ori_dds_as_gt
        stats_result_dict, stats_baseline_result_dict, costs_result_dict, costs_baseline_result_dict = \
            get_all_videos_data(file_direc, target_video_list, new_stat, dds_as_gt, \
            stats_metric, costs_metric)
        for method_name, method_result_dict in costs_baseline_result_dict.items():
            baseline_mean_cost = np.mean(list(method_result_dict.values()))
            baseline_mean_stat = np.mean(list(stats_baseline_result_dict[method_name].values()))
            if dds_as_gt and method_name == 'dds' and new_stat:
                continue
            if new_stat:
                ax2.scatter(baseline_mean_cost, baseline_mean_stat, label=f'2nd-{method_name}', marker='x')
            else:
                ax1.scatter(baseline_mean_cost, baseline_mean_stat, label=f'overall-{method_name}')
        dds_mean_cost = np.mean(list(costs_baseline_result_dict['dds'].values()))

        mean_stat_cost_pair_list = []
        mean_stat_list = []
        mean_cost_list = []
        mean_method_list = []
        for method_name, method_result_dict in costs_result_dict.items():
            method_mean_cost = np.mean(list(method_result_dict.values()))
            if compare_cost and (method_mean_cost > dds_mean_cost):
                continue
            method_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
            mean_stat_cost_pair_list.append((method_mean_stat, method_mean_cost, method_name))

        paretoPoints, dominatedPoints = simple_cull(mean_stat_cost_pair_list, dominates_stat_acc, choose_metric)
        for single_pareto_point in paretoPoints:
            mean_stat_list.append(single_pareto_point[0])
            mean_cost_list.append(single_pareto_point[1])
            mean_method_list.append(single_pareto_point[2])
        ori_cost_list1 = mean_cost_list[:]
        ori_cost_list2 = mean_cost_list[:]
        mean_stat_list, mean_cost_list = sort_by_second_list(mean_stat_list, ori_cost_list1)
        mean_method_list, mean_cost_list = sort_by_second_list(mean_method_list, ori_cost_list2)
        print('##############')
        for idx in range(len(mean_stat_list)):
            print(mean_method_list[idx], mean_stat_list[idx], mean_cost_list[idx])
        print('##############')
        
        if new_stat:
            print(mean_cost_list)
            ax2.plot(mean_cost_list, mean_stat_list, label=f'2nd-pareto', c=line_colors[0])
        else:
            print(mean_cost_list)
            ax1.plot(mean_cost_list, mean_stat_list, label=f'overall-pareto', c=line_colors[1], linestyle='--')

    ax1.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0)
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname), bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    


            
