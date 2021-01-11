import numpy as np
import os
import matplotlib.pyplot as plt
from dds_utils import (read_results_dict, Results)
from .plot_utils import (Ellipse, get_colors, read_stats_costs, confidence_ellipse, pick_list_item, \
    sort_by_second_list, find_best_stat_point, get_markers, each_trick_point, simple_cull, dominates_stat_acc, \
    get_all_videos_data, pareto_line_utils, read_logs, area_box_graph)
from .streamB_utils import (get_low_file, get_high_file, get_gt_file, get_dds_file, two_results_diff, \
    new_evaluate_all)
import pandas as pd
import shutil
import ipdb
from adjustText import adjust_text


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
    

def compare_track_trick(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count', 
        new_stat=False, dds_as_gt=False, save_direc=None, save_fname=None, plot_pareto=True,
        diff_threshold_idx=15, frame_interval_idx=17):
    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'

    fig = plt.figure()
    ax = fig.add_axes([0.12, 0.1, 0.85, 0.8])
    ax.set_ylabel(stats_metric)
    ax.set_xlabel(costs_metric)
    if new_stat:
        name_str = '2nd'
    else:
        name_str = 'overall'
    if dds_as_gt:
        name_str = name_str + '-dds'
    else:
        name_str = name_str + '-gt'
    ax.set_title(name_str)
    texts = []

    stats_result_dict, stats_baseline_result_dict, costs_result_dict, costs_baseline_result_dict = \
        get_all_videos_data(file_direc, target_video_list, new_stat, dds_as_gt, \
        stats_metric, costs_metric)

    # baseline points
    for method_name, method_result_dict in costs_baseline_result_dict.items():
        baseline_mean_cost = np.mean(list(method_result_dict.values()))
        baseline_mean_stat = np.mean(list(stats_baseline_result_dict[method_name].values()))
        if dds_as_gt and method_name == 'dds' and new_stat:
            continue
        ax.scatter(baseline_mean_cost, baseline_mean_stat)
        texts.append(plt.text(baseline_mean_cost, baseline_mean_stat, method_name))
    
    # if plotting pareto lines, store data in a list and then run pareto selection algorithm
    if plot_pareto:
        mean_stat_cost_pair_list = []
        mean_stat_list = []
        mean_cost_list = []
        
    for method_name, method_result_dict in costs_result_dict.items():
        method_mean_cost = np.mean(list(method_result_dict.values()))          
        method_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
        method_name_list = method_name.split('_')
        track_method_name = '_'.join([method_name_list[diff_threshold_idx], \
            method_name_list[frame_interval_idx]])
        ax.scatter(method_mean_cost, method_mean_stat)
        texts.append(plt.text(method_mean_cost, method_mean_stat, track_method_name))
        if plot_pareto:
            mean_stat_cost_pair_list.append((method_mean_stat, method_mean_cost))
    
    # run pareto selection
    paretoPoints, dominatedPoints = simple_cull(mean_stat_cost_pair_list, dominates_stat_acc, choose_metric)
    for single_pareto_point in paretoPoints:
        mean_stat_list.append(single_pareto_point[0])
        mean_cost_list.append(single_pareto_point[1])
    mean_stat_list, mean_cost_list = sort_by_second_list(mean_stat_list, mean_cost_list)
    print('##############')
    for idx in range(len(mean_stat_list)):
        print(mean_stat_list[idx], mean_cost_list[idx])
    print('##############')
    ax.plot(mean_cost_list, mean_stat_list)

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname))
    else:
        plt.show()
    plt.close()

            
def compare_qp_trick(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count', 
        new_stat=False, dds_as_gt=False, save_direc=None, save_fname=None,
        qp_list_idx=14, percent_list_idx=15):
    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'

    fig = plt.figure()
    ax = fig.add_axes([0.12, 0.1, 0.85, 0.8])

    stats_list = []
    costs_list = []
    new_stats_dict = {}
    for target_video_name in target_video_list:
        cur_costs_list = read_logs(os.path.join(file_direc, target_video_name, 'costs'))
        costs_list.extend(cur_costs_list)
        if new_stat:
            new_evaluate_all(os.path.join(file_direc, target_video_name), video_name=target_video_name, \
                dds_as_gt=dds_as_gt, stats_metric=stats_metric, new_stats_dict=new_stats_dict)
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'new_stats'))
            stats_list.extend(cur_stats_list)
        else:
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'stats'))
            stats_list.extend(cur_stats_list)

    qp_percent_pair_list = [('[16, 24, 26]', '[50, 75, 100]'), ('[26]', '[100]'), ('[16]', '[100]')]
    for idx, (qp_list, percent_list) in enumerate(qp_percent_pair_list):
        stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
            target_video_name=target_video_list, qp_list=qp_list, percent_list=percent_list, deal_with_comma=True)
        costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
            target_video_name=target_video_list, qp_list=qp_list, percent_list=percent_list, deal_with_comma=True)
        
        if idx == 0:
            # baseline points
            for method_name, method_result_dict in costs_baseline_result_dict.items():
                baseline_mean_cost = np.mean(list(method_result_dict.values()))
                baseline_mean_stat = np.mean(list(stats_baseline_result_dict[method_name].values()))
                if dds_as_gt and method_name == 'dds' and new_stat:
                    continue
                ax.scatter(baseline_mean_cost, baseline_mean_stat, label=method_name)
        
        mean_stat_cost_pair_list = []
        for method_name, method_result_dict in costs_result_dict.items():
            method_mean_cost = np.mean(list(method_result_dict.values()))
            method_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
            mean_stat_cost_pair_list.append((method_mean_stat, method_mean_cost, method_name))
        
        print(mean_stat_cost_pair_list)
        pareto_line_utils(ax, mean_stat_cost_pair_list, dominates_stat_acc, x_idx=1, y_idx=0, \
            choose_metric=choose_metric, plot_label=f'{qp_list}-{percent_list}')
    
    ax.set_title('qp_percent_compare')
    ax.set_ylabel(f'{stats_metric}')
    ax.set_xlabel(f'{costs_metric}')
    ax.legend(loc='best')
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname))
    else:
        plt.show()
    plt.close()    


def area_box_graph_wrapper(result_direc, save_direc, video_name, max_fid=299,
        gt_confid_thresh=0.3, compare_confid_thresh=0.5, 
        gt_max_area_thresh=0.3, compare_max_area_thresh=0.3):

    print(result_direc)
    gt_path = os.path.join(result_direc, get_gt_file(result_direc, dds_as_gt=False))
    gt_regions_dict = read_results_dict(gt_path)
    
    low_path = os.path.join(result_direc, get_low_file(result_direc))
    low_regions_dict = read_results_dict(low_path)
    high_path = os.path.join(result_direc, get_high_file(result_direc))
    high_regions_dict = read_results_dict(high_path)
    dds_path = os.path.join(result_direc, get_dds_file(result_direc))
    dds_regions_dict = read_results_dict(dds_path)
    eval_regions_dicts = [low_regions_dict, high_regions_dict, dds_regions_dict]
    eval_names = ['low', 'high', 'dds']
    
    area_all_dict = {}

    for idx, regions_dict in enumerate(eval_regions_dicts):
        area_all_dict[eval_names[idx]] = []
        fn_dict, fp_dict, _, tp_dict = two_results_diff(max_fid, gt_regions_dict, regions_dict, 
            gt_confid_thresh, compare_confid_thresh, gt_max_area_thresh, compare_max_area_thresh)
        for fid, region_list in tp_dict.items():
            for single_region in region_list:
                area_all_dict[eval_names[idx]].append(single_region.w * single_region.h)
    
    area_all_dict['gt'] = []
    for fid, region_list in gt_regions_dict.items():
        for single_region in region_list:
            area_all_dict['gt'].append(single_region.w * single_region.h)
        
    graph_title = f'{video_name}-detection_area'
    area_box_graph(area_all_dict, graph_title, save_direc, f'{graph_title}.png')
    

def compare_filter_trick(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count', 
        new_stat=False, dds_as_gt=False, save_direc=None, save_fname=None):
    
    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'

    fig = plt.figure()
    ax = fig.add_axes([0.12, 0.1, 0.85, 0.8])

    stats_list = []
    costs_list = []
    new_stats_dict = {}
    for target_video_name in target_video_list:
        cur_costs_list = read_logs(os.path.join(file_direc, target_video_name, 'costs'))
        costs_list.extend(cur_costs_list)
        if new_stat:
            new_evaluate_all(os.path.join(file_direc, target_video_name), video_name=target_video_name, \
                dds_as_gt=dds_as_gt, stats_metric=stats_metric, new_stats_dict=new_stats_dict)
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'new_stats'))
            stats_list.extend(cur_stats_list)
        else:
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'stats'))
            stats_list.extend(cur_stats_list)
    
    filter_type_list = ['dds', 'False']
    for idx, filter_type in enumerate(filter_type_list):
        stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
            target_video_name=target_video_list, filter_type=filter_type)
        costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
            target_video_name=target_video_list, filter_type=filter_type)
        
        if idx == 0:
            # baseline points
            for method_name, method_result_dict in costs_baseline_result_dict.items():
                baseline_mean_cost = np.mean(list(method_result_dict.values()))
                baseline_mean_stat = np.mean(list(stats_baseline_result_dict[method_name].values()))
                if dds_as_gt and method_name == 'dds' and new_stat:
                    continue
                ax.scatter(baseline_mean_cost, baseline_mean_stat, label=method_name)
        
        mean_stat_cost_pair_list = []
        for method_name, method_result_dict in costs_result_dict.items():
            method_mean_cost = np.mean(list(method_result_dict.values()))
            method_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
            mean_stat_cost_pair_list.append((method_mean_stat, method_mean_cost, method_name))
        
        print(mean_stat_cost_pair_list)
        pareto_line_utils(ax, mean_stat_cost_pair_list, dominates_stat_acc, x_idx=1, y_idx=0, \
            choose_metric=choose_metric, plot_label=f'filter-{filter_type}')
    
    ax.set_title('filter_type_compare')
    ax.set_ylabel(f'{stats_metric}')
    ax.set_xlabel(f'{costs_metric}')
    ax.legend(loc='best')
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname))
    else:
        plt.show()
    plt.close()    


def scatter_tricks_effect(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count', 
        new_stat=False, dds_as_gt=False, save_direc=None, save_fname=None,
        context_type_idx=0, context_val_idx=1, blank_type_idx=2, blank_val_idx=3, 
        inter_iou_idx=5, merge_iou_idx=7, resize_val_idx=9, filter_type_idx=11):
    
    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'

    fig = plt.figure()
    ax = fig.add_axes([0.12, 0.1, 0.85, 0.8])

    stats_list = []
    costs_list = []
    new_stats_dict = {}
    for target_video_name in target_video_list:
        cur_costs_list = read_logs(os.path.join(file_direc, target_video_name, 'costs'))
        costs_list.extend(cur_costs_list)
        if new_stat:
            new_evaluate_all(os.path.join(file_direc, target_video_name), video_name=target_video_name, \
                dds_as_gt=dds_as_gt, stats_metric=stats_metric, new_stats_dict=new_stats_dict)
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'new_stats'))
            stats_list.extend(cur_stats_list)
        else:
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'stats'))
            stats_list.extend(cur_stats_list)
    
    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=target_video_list)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_list)
    texts = []
    
    # baseline points
    for method_name, method_result_dict in costs_baseline_result_dict.items():
        baseline_mean_cost = np.mean(list(method_result_dict.values()))
        baseline_mean_stat = np.mean(list(stats_baseline_result_dict[method_name].values()))
        ax.scatter(baseline_mean_cost, baseline_mean_stat)
        texts.append(plt.text(baseline_mean_cost, baseline_mean_stat, method_name))

    # different tricks' points
    for method_name, method_result_dict in costs_result_dict.items():
        method_name_list = method_name.split('_')

        # no trick
        if float(method_name_list[context_val_idx]) == 0 and \
            float(method_name_list[blank_val_idx]) == 0 and \
            method_name_list[resize_val_idx] == 'no' and method_name_list[filter_type_idx] == 'False':
            no_trick_mean_cost = np.mean(list(method_result_dict.values()))
            no_trick_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
            ax.scatter(no_trick_mean_cost, no_trick_mean_stat)
            texts.append(plt.text(no_trick_mean_cost, no_trick_mean_stat, 'no_trick'))
        
        # only pad
        if float(method_name_list[context_val_idx]) != 0 and \
            float(method_name_list[blank_val_idx]) != 0 and \
            method_name_list[resize_val_idx] == 'no' and method_name_list[filter_type_idx] == 'False':
            pad_mean_cost = np.mean(list(method_result_dict.values()))
            pad_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
            ax.scatter(pad_mean_cost, pad_mean_stat)
            texts.append(plt.text(pad_mean_cost, pad_mean_stat, 'pad'))
        
        # resize
        if float(method_name_list[context_val_idx]) != 0 and \
            float(method_name_list[blank_val_idx]) != 0 and \
            method_name_list[resize_val_idx] != 'no' and method_name_list[filter_type_idx] == 'False':
            resize_mean_cost = np.mean(list(method_result_dict.values()))
            resize_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
            ax.scatter(resize_mean_cost, resize_mean_stat)
            texts.append(plt.text(resize_mean_cost, resize_mean_stat, 'resize_pad'))

        # filter
        if method_name_list[resize_val_idx] == 'no' and method_name_list[filter_type_idx] != 'False':
            filter_mean_cost = np.mean(list(method_result_dict.values()))
            filter_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
            ax.scatter(filter_mean_cost, filter_mean_stat)
            texts.append(plt.text(filter_mean_cost, filter_mean_stat, 'filter'))
            print('-------------')

        # all
        if float(method_name_list[context_val_idx]) != 0 and \
            float(method_name_list[blank_val_idx]) != 0 and \
            method_name_list[resize_val_idx] != 'no' and method_name_list[filter_type_idx] != 'False':
            all_mean_cost = np.mean(list(method_result_dict.values()))
            all_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
            ax.scatter(all_mean_cost, all_mean_stat)
            texts.append(plt.text(all_mean_cost, all_mean_stat, 'all'))
    
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname))
    else:
        plt.show()
    plt.close()


def ellipse_tricks_effect(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count', 
        new_stat=False, dds_as_gt=False, save_direc=None, save_fname=None, compare_cost=True,
        context_type_idx=0, context_val_idx=1, blank_type_idx=2, blank_val_idx=3, 
        inter_iou_idx=5, merge_iou_idx=7, resize_val_idx=9, filter_type_idx=11, 
        merge_iou=None, inter_iou=None):
    
    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])

    stats_list = []
    costs_list = []
    new_stats_dict = {}
    for target_video_name in target_video_list:
        cur_costs_list = read_logs(os.path.join(file_direc, target_video_name, 'costs'))
        costs_list.extend(cur_costs_list)
        if new_stat:
            new_evaluate_all(os.path.join(file_direc, target_video_name), video_name=target_video_name, \
                dds_as_gt=dds_as_gt, stats_metric=stats_metric, new_stats_dict=new_stats_dict)
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'new_stats'))
            stats_list.extend(cur_stats_list)
        else:
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'stats'))
            stats_list.extend(cur_stats_list)
    
    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=target_video_list)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_list)
    
    # baseline points
    for method_name, method_result_dict in costs_baseline_result_dict.items():
        baseline_mean_cost = np.mean(list(method_result_dict.values()))
        baseline_mean_stat = np.mean(list(stats_baseline_result_dict[method_name].values()))
        ax.scatter(baseline_mean_cost, baseline_mean_stat, label=method_name)
    dds_mean_cost = np.mean(list(costs_baseline_result_dict['dds'].values()))

    no_trick_costs = []
    no_trick_stats = []
    pad_costs = []
    pad_stats = []
    pad_resize_costs = []
    pad_resize_stats = []
    pad_filter_costs = []
    pad_filter_stats = []
    all_costs = []
    all_stats = []
    # different tricks' points
    for method_name, method_result_dict in costs_result_dict.items():
        method_name_list = method_name.split('_')
        method_mean_cost = np.mean(list(method_result_dict.values()))
        if compare_cost and (method_mean_cost > dds_mean_cost):
            continue
        if isinstance(merge_iou, list):
            if float(method_name_list[merge_iou_idx]) not in merge_iou:
                continue
        elif merge_iou != None and (merge_iou != float(method_name_list[merge_iou_idx])):
            continue
        if isinstance(inter_iou, list):
            if float(method_name_list[inter_iou_idx]) not in inter_iou:
                continue
        elif inter_iou != None and (inter_iou != float(method_name_list[inter_iou_idx])):
            continue
        print(method_name)

        # no trick
        if float(method_name_list[context_val_idx]) == 0 and \
            float(method_name_list[blank_val_idx]) == 0 and \
            method_name_list[resize_val_idx] == 'no' and method_name_list[filter_type_idx] == 'False':
            no_trick_mean_cost = np.mean(list(method_result_dict.values()))
            no_trick_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
            no_trick_costs.append(no_trick_mean_cost)
            no_trick_stats.append(no_trick_mean_stat)
        
        # only pad
        if float(method_name_list[context_val_idx]) != 0 and \
            float(method_name_list[blank_val_idx]) != 0 and \
            method_name_list[resize_val_idx] == 'no' and method_name_list[filter_type_idx] == 'False':
            pad_mean_cost = np.mean(list(method_result_dict.values()))
            pad_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
            pad_costs.append(pad_mean_cost)
            pad_stats.append(pad_mean_stat)
        
        # pad and resize
        if float(method_name_list[context_val_idx]) != 0 and \
            float(method_name_list[blank_val_idx]) != 0 and \
            method_name_list[resize_val_idx] != 'no' and method_name_list[filter_type_idx] == 'False':
            resize_mean_cost = np.mean(list(method_result_dict.values()))
            resize_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
            pad_resize_costs.append(resize_mean_cost)
            pad_resize_stats.append(resize_mean_stat)

        # pad and filter
        if float(method_name_list[context_val_idx]) != 0 and \
            float(method_name_list[blank_val_idx]) != 0 and \
            method_name_list[resize_val_idx] == 'no' and method_name_list[filter_type_idx] != 'False':
            filter_mean_cost = np.mean(list(method_result_dict.values()))
            filter_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
            pad_filter_costs.append(filter_mean_cost)
            pad_filter_stats.append(filter_mean_stat)

        # all
        if float(method_name_list[context_val_idx]) != 0 and \
            float(method_name_list[blank_val_idx]) != 0 and \
            method_name_list[resize_val_idx] != 'no' and method_name_list[filter_type_idx] != 'False':
            all_mean_cost = np.mean(list(method_result_dict.values()))
            all_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
            all_costs.append(all_mean_cost)
            all_stats.append(all_mean_stat)
    
    ax.scatter(no_trick_costs, no_trick_stats, label='no_trick', c='purple')
    ax.scatter(pad_costs, pad_stats, s=0.5, c='blue')
    if pad_costs and pad_stats:
        confidence_ellipse(np.array(pad_costs), np.array(pad_stats), \
            ax, n_std=2, label='only_padding', 
            alpha=0.5, edgecolor='blue', facecolor='lightblue', linestyle=':')
    ax.scatter(pad_resize_costs, pad_resize_stats, s=0.5, c='fuchsia')
    if pad_resize_costs and pad_resize_stats:
        confidence_ellipse(np.array(pad_resize_costs), np.array(pad_resize_stats), \
            ax, n_std=2, label='pad_resize', 
            alpha=0.5, edgecolor='fuchsia', facecolor='pink', linestyle='--')
    ax.scatter(pad_filter_costs, pad_filter_stats, s=0.5, c='firebrick')
    if pad_filter_costs and pad_filter_stats:
        confidence_ellipse(np.array(pad_filter_costs), np.array(pad_filter_stats), \
            ax, n_std=2, label='pad_filter', 
            alpha=0.5, edgecolor='firebrick', facecolor='indianred', linestyle='-.')
    ax.scatter(all_costs, all_stats, s=0.5, c='lightgreen')
    if all_costs and all_stats:
        confidence_ellipse(np.array(all_costs), np.array(all_stats), \
            ax, n_std=2, label='pad_resize_filter', 
            alpha=0.5, edgecolor='lightgreen', facecolor='darkseagreen')
    
    plt.title("tricks_comparison")
    ax.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0)
    ax.set_xlabel(f'{costs_metric}')
    ax.set_ylabel(f'{stats_metric}')
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname))
    else:
        plt.show()
    plt.close()


def compare_merge_iou(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count', 
        new_stat=False, dds_as_gt=False, save_direc=None, save_fname=None, 
        merge_iou_idx=7, merge_iou_list=[0.0, 1.0]):
    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    ax1.set_ylabel(f'{stats_metric}')
    ax2 = ax1.twinx()
    ax2.set_ylabel(f'{costs_metric}')

    stats_list = []
    costs_list = []
    new_stats_dict = {}
    for target_video_name in target_video_list:
        cur_costs_list = read_logs(os.path.join(file_direc, target_video_name, 'costs'))
        costs_list.extend(cur_costs_list)
        if new_stat:
            new_evaluate_all(os.path.join(file_direc, target_video_name), video_name=target_video_name, \
                dds_as_gt=dds_as_gt, stats_metric=stats_metric, new_stats_dict=new_stats_dict)
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'new_stats'))
            stats_list.extend(cur_stats_list)
        else:
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'stats'))
            stats_list.extend(cur_stats_list)
    
    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=target_video_list)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_list)

    dds_mean_cost = np.mean(list(costs_baseline_result_dict['dds'].values()))

    iou_cnt = len(merge_iou_list)
    kept_method_name_list = []
    removed_method_name_list = []
    iou_all_costs = []
    iou_all_stats = []
    for i in range(iou_cnt):
        iou_all_costs.append([])
        iou_all_stats.append([])
 
    for method_name, method_result_dict in costs_result_dict.items():
        method_name_list = method_name.split('_')
        method_mean_cost = np.mean(list(method_result_dict.values()))
        cur_iou = float(method_name_list[merge_iou_idx])
        if cur_iou not in merge_iou_list:
            continue
        method_name_list[merge_iou_idx] = 'all'
        method_key = '_'.join(method_name_list)
        method_mean_cost = np.mean(list(method_result_dict.values()))
        if method_mean_cost > dds_mean_cost:
            if method_key not in removed_method_name_list:
                removed_method_name_list.append(method_key)
            if method_key in kept_method_name_list:
                kept_method_name_list.remove(method_key)
        elif (method_key not in kept_method_name_list) and (method_key not in removed_method_name_list):
            kept_method_name_list.append(method_key)
    print(kept_method_name_list)

    for method_key in kept_method_name_list:
        find_all = True
        for i in range(iou_cnt):
            method_name_list = method_key.split('_')
            method_name_list[merge_iou_idx] = str(merge_iou_list[i])
            method_name = '_'.join(method_name_list)
            if method_name not in costs_result_dict.keys():
                print(method_name)
                find_all = False
                break
        if not find_all:
            continue

        for i in range(iou_cnt):
            method_name_list = method_key.split('_')
            method_name_list[merge_iou_idx] = str(merge_iou_list[i])
            method_name = '_'.join(method_name_list)
            method_mean_cost = np.mean(list(costs_result_dict[method_name].values()))
            method_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
            iou_all_costs[i].append(method_mean_cost)
            iou_all_stats[i].append(method_mean_stat)
    
    unorder_costs = iou_all_costs[-1][:]
    for i in range(iou_cnt):
        tmp_unorder_costs = unorder_costs[:]
        iou_all_costs[i], _ = sort_by_second_list(iou_all_costs[i], tmp_unorder_costs)
        tmp_unorder_costs = unorder_costs[:]
        iou_all_stats[i], _ = sort_by_second_list(iou_all_stats[i], tmp_unorder_costs)
        ax1.plot(iou_all_stats[i], label=f"{stats_metric}_{merge_iou_list[i]}")
        ax2.plot(iou_all_costs[i], label=f"{costs_metric}_{merge_iou_list[i]}", linestyle='--')    
        
    plt.title("merge iou comparison")
    ax1.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname))
    else:
        plt.show()
    plt.close()
    
        
def compare_inter_iou(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count', 
        new_stat=False, dds_as_gt=False, save_direc=None, save_fname=None, 
        inter_iou_idx=5, inter_iou_list=[0.0, 0.2]):
    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    ax.set_xlabel(f'{costs_metric}')
    ax.set_ylabel(f'{stats_metric}')

    stats_list = []
    costs_list = []
    new_stats_dict = {}
    for target_video_name in target_video_list:
        cur_costs_list = read_logs(os.path.join(file_direc, target_video_name, 'costs'))
        costs_list.extend(cur_costs_list)
        if new_stat:
            new_evaluate_all(os.path.join(file_direc, target_video_name), video_name=target_video_name, \
                dds_as_gt=dds_as_gt, stats_metric=stats_metric, new_stats_dict=new_stats_dict)
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'new_stats'))
            stats_list.extend(cur_stats_list)
        else:
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'stats'))
            stats_list.extend(cur_stats_list)
    
    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=target_video_list)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_list)

    # baseline points
    for method_name, method_result_dict in costs_baseline_result_dict.items():
        baseline_mean_cost = np.mean(list(method_result_dict.values()))
        baseline_mean_stat = np.mean(list(stats_baseline_result_dict[method_name].values()))
        ax.scatter(baseline_mean_cost, baseline_mean_stat, label=method_name)
    dds_mean_cost = np.mean(list(costs_baseline_result_dict['dds'].values()))
    dds_mean_cost = np.mean(list(costs_baseline_result_dict['dds'].values()))

    iou_cnt = len(inter_iou_list)
    kept_method_name_list = []
    removed_method_name_list = []
    iou_all_costs = []
    iou_all_stats = []
    for i in range(iou_cnt):
        iou_all_costs.append([])
        iou_all_stats.append([])
 
    for method_name, method_result_dict in costs_result_dict.items():
        method_name_list = method_name.split('_')
        method_mean_cost = np.mean(list(method_result_dict.values()))
        cur_iou = float(method_name_list[inter_iou_idx])
        if cur_iou not in inter_iou_list:
            continue
        method_name_list[inter_iou_idx] = 'all'
        method_key = '_'.join(method_name_list)
        method_mean_cost = np.mean(list(method_result_dict.values()))
        if method_mean_cost > dds_mean_cost:
            if method_key not in removed_method_name_list:
                removed_method_name_list.append(method_key)
            if method_key in kept_method_name_list:
                kept_method_name_list.remove(method_key)
        elif (method_key not in kept_method_name_list) and (method_key not in removed_method_name_list):
            kept_method_name_list.append(method_key)
    print(kept_method_name_list)

    for method_key in kept_method_name_list:
        find_all = True
        for i in range(iou_cnt):
            method_name_list = method_key.split('_')
            method_name_list[inter_iou_idx] = str(inter_iou_list[i])
            method_name = '_'.join(method_name_list)
            if method_name not in costs_result_dict.keys():
                print(method_name)
                find_all = False
                break
        if not find_all:
            continue

        for i in range(iou_cnt):
            method_name_list = method_key.split('_')
            method_name_list[inter_iou_idx] = str(inter_iou_list[i])
            method_name = '_'.join(method_name_list)
            method_mean_cost = np.mean(list(costs_result_dict[method_name].values()))
            method_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
            iou_all_costs[i].append(method_mean_cost)
            iou_all_stats[i].append(method_mean_stat)
    
    unorder_costs = iou_all_costs[-1][:]
    for i in range(iou_cnt):
        tmp_unorder_costs = unorder_costs[:]
        iou_all_costs[i], _ = sort_by_second_list(iou_all_costs[i], tmp_unorder_costs)
        tmp_unorder_costs = unorder_costs[:]
        iou_all_stats[i], _ = sort_by_second_list(iou_all_stats[i], tmp_unorder_costs)
        ax.plot(iou_all_costs[i], iou_all_stats[i], label=f'inter_{inter_iou_list[i]}')  
        
    plt.title("intersection iou comparison")
    ax.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname))
    else:
        plt.show()
    plt.close()


def pareto_all_methods(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count', 
        new_stat=False, dds_as_gt=False, save_direc=None, save_fname=None, fig_title=None, compare_cost=True,
        context_type_idx=0, context_val_idx=1, blank_type_idx=2, blank_val_idx=3, 
        inter_iou_idx=5, merge_iou_idx=7, resize_val_idx=9, filter_type_idx=11, 
        context_type=None, blank_type=None, merge_iou=None, inter_iou=None):
    
    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])

    stats_list = []
    costs_list = []
    new_stats_dict = {}
    for target_video_name in target_video_list:
        cur_costs_list = read_logs(os.path.join(file_direc, target_video_name, 'costs'))
        costs_list.extend(cur_costs_list)
        if new_stat:
            new_evaluate_all(os.path.join(file_direc, target_video_name), video_name=target_video_name, \
                dds_as_gt=dds_as_gt, stats_metric=stats_metric, new_stats_dict=new_stats_dict)
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'new_stats'))
            stats_list.extend(cur_stats_list)
        else:
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'stats'))
            stats_list.extend(cur_stats_list)
    
    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=target_video_list, context_type=context_type, blank_type=blank_type, 
        merge_iou=merge_iou, inter_iou=inter_iou)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_list, context_type=context_type, blank_type=blank_type, 
        merge_iou=merge_iou, inter_iou=inter_iou)
    texts = []
    
    # baseline points
    for method_name, method_result_dict in costs_baseline_result_dict.items():
        baseline_mean_cost = np.mean(list(method_result_dict.values()))
        baseline_mean_stat = np.mean(list(stats_baseline_result_dict[method_name].values()))
        ax.scatter(baseline_mean_cost, baseline_mean_stat)
        texts.append(plt.text(baseline_mean_cost, baseline_mean_stat, method_name))
    dds_mean_cost = np.mean(list(costs_baseline_result_dict['dds'].values()))

    mean_stat_cost_pair_list = []
    for method_name, method_result_dict in costs_result_dict.items():
        method_name_list = method_name.split('_')
        method_mean_cost = np.mean(list(method_result_dict.values()))
        if compare_cost and (method_mean_cost > dds_mean_cost):
            continue
        new_method_list = method_name_list[context_type_idx:blank_val_idx+1]
        # new_method_list.append(method_name_list[inter_iou_idx])
        # new_method_list.append(method_name_list[merge_iou_idx])
        new_method_list.append(method_name_list[resize_val_idx])
        new_method_list.append(method_name_list[filter_type_idx])
        new_method_name = '_'.join(new_method_list)

        method_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
        mean_stat_cost_pair_list.append((method_mean_stat, method_mean_cost, new_method_name))
    
    print(mean_stat_cost_pair_list)
    x_list, y_list, method_list = pareto_line_utils(
        ax, mean_stat_cost_pair_list, dominates_stat_acc, x_idx=1, y_idx=0, \
        choose_metric=choose_metric)
    for i in range(len(x_list)):
        texts.append(plt.text(x_list[i], y_list[i], method_list[i], fontsize='x-small'))
    
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    if fig_title:
        ax.set_title(fig_title)
    else:
        ax.set_title('methods pareto line')
    ax.set_ylabel(f'{stats_metric}')
    ax.set_xlabel(f'{costs_metric}')
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname))
    else:
        plt.show()
    plt.close()    


def compare_resize_region_area(file_direc, gt_fname, res_fnames, 
        gt_confid_thresh=0.3, gt_max_area_thresh=0.3, 
        compare_confid_thresh=0.5, compare_max_area_thresh=0.3, 
        context_type_idx=3, blank_val_idx=6, resize_val_idx=12):

    gt_results_dict = read_results_dict(os.path.join(file_direc, gt_fname))
    max_fid = max(list(gt_results_dict.keys()))
    area_dict = {}
    area_boundary = []

    for fname in res_fnames:
        compare_results_dict = read_results_dict(os.path.join(file_direc, fname))
        _, _, _, tp_dict = two_results_diff(max_fid, gt_results_dict, compare_results_dict, \
            gt_confid_thresh, compare_confid_thresh, gt_max_area_thresh, compare_max_area_thresh)
        method_name_list = fname.split('_')
        resize_method = '_'.join(method_name_list[resize_val_idx-1:resize_val_idx+1])
        pad_method = '_'.join(method_name_list[context_type_idx:blank_val_idx+1])
        area_dict[resize_method] = []
        if method_name_list[resize_val_idx] != 'no':
            area_boundary.append(float(method_name_list[resize_val_idx]))

        for fid, region_list in tp_dict.items():
            for single_region in region_list:
                area_dict[resize_method].append(single_region.w * single_region.h)
    area_boundary.sort()

    area_count = {}
    for resize_method, area_list in area_dict.items():
        area_count[resize_method] = np.zeros(len(area_boundary) + 1)
        for single_area in area_list:
            if single_area <= area_boundary[0]:
                area_count[resize_method][0] += 1
            elif single_area > area_boundary[-1]:
                area_count[resize_method][-1] += 1
            for idx in range(len(area_boundary)-1):
                if area_boundary[idx] < single_area < area_boundary[idx+1]:
                    area_count[resize_method][idx+1] += 1
    
    fig = plt.figure()
    plt.grid(linestyle=':', axis='y')
    plt.xlabel("resize methods")
    plt.ylabel("area count")
    plt.title(f"{pad_method} region area comparison")

    bar_width = round(1/(len(area_boundary) + 1), 1)
    x = np.arange(len( list(area_count.keys()) ))
    xgroup_labels = []
    x_loc = []
    y_data = []
    for i in range(len(area_boundary) + 1):
        x_loc.append([])
        y_data.append([])

    for i, resize_method in enumerate(list(area_count.keys())):
        xgroup_labels.append(resize_method)
        for j, area_bin_cnt in enumerate(area_count[resize_method]):
            x_loc[j].append( i + (j - (len(area_boundary)/2))*bar_width )
            y_data[j].append(area_bin_cnt)
    
    for i in range(len(area_boundary) + 1):
        if i==0:
            area_label = f'(0,{area_boundary[0]}]'
        elif i==len(area_boundary):
            area_label = f'({area_boundary[-1]},1]'
        else:
            area_label = f'({area_boundary[i-1]}, {area_boundary[i]}]'
        plt.bar(x_loc[i], y_data[i], bar_width, label=area_label)

    plt.xticks(x, list(area_count.keys()))
    plt.legend()
    plt.show()


def more_cost_method(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count'):
    
    costs_list = []
    stats_list = []
    for target_video_name in target_video_list:
        cur_costs_list = read_logs(os.path.join(file_direc, target_video_name, 'costs'))
        costs_list.extend(cur_costs_list)
        cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'stats'))
        stats_list.extend(cur_stats_list)
        
    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=target_video_list)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_list)       
    
    dds_mean_cost = np.mean(list(costs_baseline_result_dict['dds'].values()))
    dds_mean_stat = np.mean(list(stats_baseline_result_dict['dds'].values()))
    print(dds_mean_cost, dds_mean_stat)
    for method_name, method_result_dict in costs_result_dict.items():
        method_name_list = method_name.split('_')
        method_mean_cost = np.mean(list(method_result_dict.values()))
        method_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
        if method_mean_cost <= dds_mean_cost:
            print(method_name, method_mean_cost, method_mean_stat)


def add_trick_one_by_one(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count', 
        new_stat=False, dds_as_gt=False, save_direc=None, save_fname=None, compare_cost=True,
        context_type_idx=0, context_val_idx=1, blank_type_idx=2, blank_val_idx=3, 
        inter_iou_idx=5, merge_iou_idx=7, resize_val_idx=9, filter_type_idx=11, 
        context_type=None, context_val=None, blank_type=None, blank_val=None,
        merge_iou=None, inter_iou=None, resize_val="0.005", filter_type="dds"):
    
    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])

    stats_list = []
    costs_list = []
    new_stats_dict = {}
    for target_video_name in target_video_list:
        cur_costs_list = read_logs(os.path.join(file_direc, target_video_name, 'costs'))
        costs_list.extend(cur_costs_list)
        if new_stat:
            new_evaluate_all(os.path.join(file_direc, target_video_name), video_name=target_video_name, \
                dds_as_gt=dds_as_gt, stats_metric=stats_metric, new_stats_dict=new_stats_dict)
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'new_stats'))
            stats_list.extend(cur_stats_list)
        else:
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'stats'))
            stats_list.extend(cur_stats_list)
    
    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=target_video_list, 
        context_type=context_type, context_val=context_val, blank_type=blank_type, blank_val=blank_val, 
        merge_iou=merge_iou, inter_iou=inter_iou)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_list, 
        context_type=context_type, context_val=context_val, blank_type=blank_type, blank_val=blank_val, 
        merge_iou=merge_iou, inter_iou=inter_iou)
    no_trick_stats, _ = pick_list_item(stats_list, stats_metric,
        target_video_name=target_video_list, 
        context_type="whole", context_val=0.0, blank_type="whole", blank_val=0.0, 
        merge_iou=merge_iou, inter_iou=inter_iou, resize_val="no", filter_type="False")
    no_trick_costs, _ = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_list, 
        context_type="whole", context_val=0.0, blank_type="whole", blank_val=0.0, 
        merge_iou=merge_iou, inter_iou=inter_iou, resize_val="no", filter_type="False")
    texts = []
    
    # baseline points
    for method_name, method_result_dict in costs_baseline_result_dict.items():
        baseline_mean_cost = np.mean(list(method_result_dict.values()))
        baseline_mean_stat = np.mean(list(stats_baseline_result_dict[method_name].values()))
        ax.scatter(baseline_mean_cost, baseline_mean_stat)
        texts.append(plt.text(baseline_mean_cost, baseline_mean_stat, method_name))
    dds_mean_cost = np.mean(list(costs_baseline_result_dict['dds'].values()))

    # no trick points
    print(no_trick_costs.keys())
    for method_name, method_result_dict in no_trick_costs.items():
        method_mean_cost = np.mean(list(method_result_dict.values()))
        method_mean_stat = np.mean(list(no_trick_stats[method_name].values()))
        ax.scatter(method_mean_cost, method_mean_stat)
        texts.append(plt.text(method_mean_cost, method_mean_stat, "no trick"))
    
    # Add trick one by one
    print(costs_result_dict.keys())
    for method_name, method_result_dict in costs_result_dict.items():
        method_name_list = method_name.split("_")
        method_mean_cost = np.mean(list(method_result_dict.values()))
        method_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
        if method_name_list[resize_val_idx] == "no" and \
            method_name_list[filter_type_idx] == "False":
            ax.scatter(method_mean_cost, method_mean_stat)
            texts.append(plt.text(method_mean_cost, method_mean_stat, "only_pad"))
        if method_name_list[resize_val_idx] == resize_val and \
            method_name_list[filter_type_idx] == "False":
            ax.scatter(method_mean_cost, method_mean_stat)
            texts.append(plt.text(method_mean_cost, method_mean_stat, "pad_resize"))
        if method_name_list[resize_val_idx] == resize_val and \
            method_name_list[filter_type_idx] == filter_type:
            print(method_name)
            ax.scatter(method_mean_cost, method_mean_stat)
            texts.append(plt.text(method_mean_cost, method_mean_stat, "pad_resize_filter"))
    
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    ax.set_title("tricks point")
    ax.set_ylabel(f'{stats_metric}')
    ax.set_xlabel(f'{costs_metric}')
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname))
    else:
        plt.show()
    plt.close()    


def get_top_stats(file_direc, target_video_list, top_stats_count, 
        stats_metric='F1', costs_metric='frame-count', 
        save_direc=None, save_fname=None, 
        context_type_idx=0, context_val_idx=1, blank_type_idx=2, blank_val_idx=3, 
        inter_iou_idx=5, merge_iou_idx=7, resize_val_idx=9, filter_type_idx=11, 
        context_type=None, context_val=None, blank_type=None, blank_val=None,
        merge_iou=None, inter_iou=None, resize_val=None, filter_type=None):

    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'
    
    stats_list = []
    costs_list = []

    for target_video_name in target_video_list:
        # cur_costs_list = read_logs(os.path.join(file_direc, target_video_name, 'costs'))
        # costs_list.extend(cur_costs_list)
        cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'stats'))
        stats_list.extend(cur_stats_list)
    
    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=target_video_list, 
        context_type=context_type, context_val=context_val, blank_type=blank_type, blank_val=blank_val, 
        merge_iou=merge_iou, inter_iou=inter_iou, resize_val=resize_val, filter_type=filter_type)
    # costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
    #     target_video_name=target_video_list, 
    #     context_type=context_type, context_val=context_val, blank_type=blank_type, blank_val=blank_val, 
    #     merge_iou=merge_iou, inter_iou=inter_iou, resize_val=resize_val, filter_type=filter_type)

    top_stats_method_dict = {}
    top_stats_dict = {}
    for target_video_name in target_video_list:
        top_stats_method_dict[target_video_name] = []
        top_stats_dict[target_video_name] = []
    for method_name, method_result_dict in stats_result_dict.items():
        for target_video_name in target_video_list:
            if target_video_name in method_result_dict.keys():
                top_stats_method_dict[target_video_name].append(method_name)
                top_stats_dict[target_video_name].append(method_result_dict[target_video_name])
    
    # Sort method stats for each video
    for target_video_name in target_video_list:
        cur_method_list = top_stats_method_dict[target_video_name]
        cur_stats_list = top_stats_dict[target_video_name]
        cur_method_list, cur_stats_list = \
            sort_by_second_list(cur_method_list, cur_stats_list)
        selected_method_count = min(len(cur_method_list), top_stats_count)
        if choose_metric == "max":
            cur_method_list = cur_method_list[::-1]
            cur_stats_list = cur_stats_list[::-1]
        top_stats_method_dict[target_video_name] = cur_method_list[0:selected_method_count]
        top_stats_dict[target_video_name] = cur_stats_list[0:selected_method_count]
        print(target_video_name)
        print(top_stats_method_dict[target_video_name])
        print(top_stats_dict[target_video_name])
    
    return top_stats_method_dict, top_stats_dict
        

def get_top_stats_costs_ratio(file_direc, target_video_list, top_ratio_count, 
        stats_metric='F1', costs_metric='frame-count', 
        save_direc=None, save_fname=None, low_qp=36, 
        context_type_idx=0, context_val_idx=1, blank_type_idx=2, blank_val_idx=3, 
        inter_iou_idx=5, merge_iou_idx=7, resize_val_idx=9, filter_type_idx=11, 
        context_type=None, context_val=None, blank_type=None, blank_val=None,
        merge_iou=None, inter_iou=None, resize_val=None, filter_type=None):

    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'
    
    stats_list = []
    costs_list = []

    for target_video_name in target_video_list:
        cur_costs_list = read_logs(os.path.join(file_direc, target_video_name, 'costs'))
        costs_list.extend(cur_costs_list)
        cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'stats'))
        stats_list.extend(cur_stats_list)
    
    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=target_video_list, 
        context_type=context_type, context_val=context_val, blank_type=blank_type, blank_val=blank_val, 
        merge_iou=merge_iou, inter_iou=inter_iou, resize_val=resize_val, filter_type=filter_type)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_list, 
        context_type=context_type, context_val=context_val, blank_type=blank_type, blank_val=blank_val, 
        merge_iou=merge_iou, inter_iou=inter_iou, resize_val=resize_val, filter_type=filter_type)
    
    low_stats_dict = {}
    for method_name, method_result_dict in stats_baseline_result_dict.items():
        if str(low_qp) in method_name:
            print(method_name)
            for target_video_name in target_video_list:
                low_stats_dict[target_video_name] = method_result_dict[target_video_name]
    dds_stats_dict = {}
    for target_video_name in target_video_list:
        dds_stats_dict[target_video_name] = stats_baseline_result_dict['dds'][target_video_name]
    print(dds_stats_dict)

    top_ratio_method_dict = {}
    top_ratio_dict = {}
    top_stats_dict = {}
    top_costs_dict = {}
    for target_video_name in target_video_list:
        top_ratio_method_dict[target_video_name] = []
        top_ratio_dict[target_video_name] = []
        top_stats_dict[target_video_name] = []
        top_costs_dict[target_video_name] = []
    for method_name, method_result_dict in stats_result_dict.items():
        for target_video_name in target_video_list:
            if target_video_name in method_result_dict.keys():
                cur_stat = method_result_dict[target_video_name]
                if cur_stat < dds_stats_dict[target_video_name]:
                    continue
                top_ratio_method_dict[target_video_name].append(method_name)
                top_stats_dict[target_video_name].append(cur_stat)
                cur_stat -= low_stats_dict[target_video_name]
                cur_cost = costs_result_dict[method_name][target_video_name]
                top_costs_dict[target_video_name].append(cur_cost)
                top_ratio_dict[target_video_name].append(cur_stat / cur_cost)
    
    # Sort method ratio for each video
    for target_video_name in target_video_list:
        cur_method_list = top_ratio_method_dict[target_video_name]
        cur_ratio_list = top_ratio_dict[target_video_name]
        cur_stats_list = top_stats_dict[target_video_name]
        cur_costs_list = top_costs_dict[target_video_name]

        tmp_ratio_list = cur_ratio_list[:]
        cur_method_list, _ = \
            sort_by_second_list(cur_method_list, tmp_ratio_list)
        tmp_ratio_list = cur_ratio_list[:]
        cur_stats_list, _ = \
            sort_by_second_list(cur_stats_list, tmp_ratio_list)
        cur_costs_list, cur_ratio_list = \
            sort_by_second_list(cur_costs_list, cur_ratio_list)

        selected_method_count = min(len(cur_method_list), top_ratio_count)
        if choose_metric == "max":
            cur_method_list = cur_method_list[::-1]
            cur_ratio_list = cur_ratio_list[::-1]
            cur_stats_list = cur_stats_list[::-1]
            cur_costs_list = cur_costs_list[::-1]
        top_ratio_method_dict[target_video_name] = cur_method_list[0:selected_method_count]
        top_ratio_dict[target_video_name] = cur_ratio_list[0:selected_method_count]
        top_stats_dict[target_video_name] = cur_stats_list[0:selected_method_count]
        top_costs_dict[target_video_name] = cur_costs_list[0:selected_method_count]

        print(target_video_name)
        print(top_ratio_method_dict[target_video_name])
        print(top_ratio_dict[target_video_name])
        print(top_stats_dict[target_video_name])
        print(top_costs_dict[target_video_name])
    return top_ratio_method_dict, top_ratio_dict


def get_top_stats_ratio(file_direc, target_video_list, top_stats_count, top_ratio_count, 
        stats_metric='F1', costs_metric='frame-count', 
        save_direc=None, save_fname=None, low_qp=36, 
        context_type_idx=0, context_val_idx=1, blank_type_idx=2, blank_val_idx=3, 
        inter_iou_idx=5, merge_iou_idx=7, resize_val_idx=9, filter_type_idx=11, 
        context_type=None, context_val=None, blank_type=None, blank_val=None,
        merge_iou=None, inter_iou=None, resize_val=None, filter_type=None):

    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'
    
    stats_list = []
    costs_list = []

    for target_video_name in target_video_list:
        cur_costs_list = read_logs(os.path.join(file_direc, target_video_name, 'costs'))
        costs_list.extend(cur_costs_list)
        cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'stats'))
        stats_list.extend(cur_stats_list)
    
    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=target_video_list, 
        context_type=context_type, context_val=context_val, blank_type=blank_type, blank_val=blank_val, 
        merge_iou=merge_iou, inter_iou=inter_iou, resize_val=resize_val, filter_type=filter_type)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_list, 
        context_type=context_type, context_val=context_val, blank_type=blank_type, blank_val=blank_val, 
        merge_iou=merge_iou, inter_iou=inter_iou, resize_val=resize_val, filter_type=filter_type)
    
    low_stats_dict = {}
    # for method_name, method_result_dict in stats_baseline_result_dict.items():
    #     if str(low_qp) in method_name:
    #         print(method_name)
    #         for target_video_name in target_video_list:
    #             low_stats_dict[target_video_name] = method_result_dict[target_video_name]

    top_ratio_method_dict = {}
    top_ratio_dict = {}
    top_stats_dict = {}
    top_costs_dict = {}
    for target_video_name in target_video_list:
        top_ratio_method_dict[target_video_name] = []
        top_ratio_dict[target_video_name] = []
        top_stats_dict[target_video_name] = []
        top_costs_dict[target_video_name] = []
    for method_name, method_result_dict in stats_result_dict.items():
        for target_video_name in target_video_list:
            if target_video_name in method_result_dict.keys():
                cur_stat = method_result_dict[target_video_name]
                if cur_stat < low_stats_dict[target_video_name]:
                    continue
                top_ratio_method_dict[target_video_name].append(method_name)
                top_stats_dict[target_video_name].append(cur_stat)
                cur_stat -= low_stats_dict[target_video_name]
                cur_cost = costs_result_dict[method_name][target_video_name]
                top_costs_dict[target_video_name].append(cur_cost)
                top_ratio_dict[target_video_name].append(cur_stat / cur_cost)
    
    # First select top stats
    for target_video_name in target_video_list:
        cur_method_list = top_ratio_method_dict[target_video_name]
        cur_ratio_list = top_ratio_dict[target_video_name]
        cur_stats_list = top_stats_dict[target_video_name]
        cur_costs_list = top_costs_dict[target_video_name]

        tmp_stats_list = cur_stats_list[:]
        cur_method_list, _ = \
            sort_by_second_list(cur_method_list, tmp_stats_list)
        tmp_stats_list = cur_stats_list[:]
        cur_ratio_list, _ = \
            sort_by_second_list(cur_ratio_list, tmp_stats_list)
        cur_costs_list, cur_stats_list = \
            sort_by_second_list(cur_costs_list, cur_stats_list)
        
        selected_method_count = min(len(cur_method_list), top_stats_count)
        if choose_metric == "max":
            cur_method_list = cur_method_list[::-1]
            cur_ratio_list = cur_ratio_list[::-1]
            cur_stats_list = cur_stats_list[::-1]
            cur_costs_list = cur_costs_list[::-1]
        top_ratio_method_dict[target_video_name] = cur_method_list[0:selected_method_count]
        top_ratio_dict[target_video_name] = cur_ratio_list[0:selected_method_count]
        top_stats_dict[target_video_name] = cur_stats_list[0:selected_method_count]
        top_costs_dict[target_video_name] = cur_costs_list[0:selected_method_count]
    
    # Then select top ratio ratio
    for target_video_name in target_video_list:
        cur_method_list = top_ratio_method_dict[target_video_name]
        cur_ratio_list = top_ratio_dict[target_video_name]
        cur_stats_list = top_stats_dict[target_video_name]
        cur_costs_list = top_costs_dict[target_video_name]

        tmp_ratio_list = cur_ratio_list[:]
        cur_method_list, _ = \
            sort_by_second_list(cur_method_list, tmp_ratio_list)
        tmp_ratio_list = cur_ratio_list[:]
        cur_stats_list, _ = \
            sort_by_second_list(cur_stats_list, tmp_ratio_list)
        cur_costs_list, cur_ratio_list = \
            sort_by_second_list(cur_costs_list, cur_ratio_list)

        selected_method_count = min(len(cur_method_list), top_ratio_count)
        if choose_metric == "max":
            cur_method_list = cur_method_list[::-1]
            cur_ratio_list = cur_ratio_list[::-1]
            cur_stats_list = cur_stats_list[::-1]
            cur_costs_list = cur_costs_list[::-1]
        top_ratio_method_dict[target_video_name] = cur_method_list[0:selected_method_count]
        top_ratio_dict[target_video_name] = cur_ratio_list[0:selected_method_count]
        top_stats_dict[target_video_name] = cur_stats_list[0:selected_method_count]
        top_costs_dict[target_video_name] = cur_costs_list[0:selected_method_count]

        print(target_video_name)
        print(top_ratio_method_dict[target_video_name])
        print(top_ratio_dict[target_video_name])
        print(top_stats_dict[target_video_name])
        print(top_costs_dict[target_video_name])


def get_method_rank(target_video_list, top_method_dict, top_metric_dict=None):
    overall_method_rank = {}
    for target_video_name in target_video_list:
        for top_method in top_method_dict[target_video_name]:
            if top_method not in overall_method_rank.keys():
                overall_method_rank[top_method] = 0
            overall_method_rank[top_method] += 1
    sorted_method_rank = sorted(overall_method_rank.items(), key = lambda x:x[1], reverse = True)
    print(sorted_method_rank)
