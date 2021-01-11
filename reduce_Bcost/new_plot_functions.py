import numpy as np
import os
import matplotlib.pyplot as plt
from dds_utils import (read_results_dict, Results)
from .plot_utils import (Ellipse, get_colors, read_stats_costs, confidence_ellipse, pick_list_item, \
    sort_by_second_list, find_best_stat_point, get_markers, simple_cull, dominates_stat_acc, \
    get_all_videos_data, pareto_line_utils, read_logs, area_box_graph)
from .streamB_utils import (two_results_diff)
import pandas as pd
import shutil
import ipdb
from adjustText import adjust_text


def single_pareto_trick(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count', 
        ax=None, compare_cost=True, plot_baseline_point=False, plot_baseline_line=False, 
        need_method_text=False, text_color="r",
        context_type_idx=0, context_val_idx=1, blank_type_idx=2, blank_val_idx=3, 
        inter_iou_idx=5, merge_iou_idx=7, resize_val_idx=9, filter_type_idx=11, 
        context_type=None, blank_type=None, blank_val=None, 
        merge_iou=None, inter_iou=None, filter_type=None):
    
    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'
    if not ax:
        return

    stats_list = []
    costs_list = []
    for target_video_name in target_video_list:
        cur_costs_list = read_logs(os.path.join(file_direc, target_video_name, 'costs'))
        costs_list.extend(cur_costs_list)
        cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'stats'))
        stats_list.extend(cur_stats_list)
    
    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=target_video_list, 
        context_type=context_type, blank_type=blank_type, blank_val=blank_val,
        merge_iou=merge_iou, inter_iou=inter_iou, filter_type=filter_type)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_list, 
        context_type=context_type, blank_type=blank_type, blank_val=blank_val, 
        merge_iou=merge_iou, inter_iou=inter_iou, filter_type=filter_type)
    texts = []

    # Baseline points
    if plot_baseline_point:
        for method_name, method_result_dict in costs_baseline_result_dict.items():
            baseline_mean_cost = np.mean(list(method_result_dict.values()))
            baseline_mean_stat = np.mean(list(stats_baseline_result_dict[method_name].values()))
            ax.scatter(baseline_mean_cost, baseline_mean_stat, label=method_name)
            if plot_baseline_line:
                ax.hlines(baseline_mean_stat, 0, 1, transform=ax.get_yaxis_transform(), \
                    linestyles='dashed', linewidths=0.5, colors='pink')
            texts.append(plt.text(baseline_mean_cost, baseline_mean_stat, method_name))
    dds_mean_cost = np.mean(list(costs_baseline_result_dict['dds'].values()))

    # Method points
    mean_stat_cost_pair_list = []
    for method_name, method_result_dict in costs_result_dict.items():
        method_name_list = method_name.split('_')
        method_mean_cost = np.mean(list(method_result_dict.values()))
        if compare_cost and (method_mean_cost > dds_mean_cost):
            continue
        new_method_list = method_name_list[context_type_idx:blank_val_idx+1]
        new_method_list.append(method_name_list[resize_val_idx])
        new_method_list.append(method_name_list[filter_type_idx])
        new_method_name = '_'.join(new_method_list)

        method_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
        mean_stat_cost_pair_list.append((method_mean_stat, method_mean_cost, new_method_name))

    plot_label = None
    if not need_method_text:
        if blank_val != 0.0:
            plot_label = f"{context_type}_{blank_type}"
        else:
            plot_label = f"{context_type}_no"

    x_list, y_list, method_list = pareto_line_utils(
        ax, mean_stat_cost_pair_list, dominates_stat_acc, x_idx=1, y_idx=0, \
        choose_metric=choose_metric, plot_label=plot_label)
    if need_method_text:
        for i in range(len(x_list)):
            texts.append(plt.text(x_list[i], y_list[i], method_list[i], \
                fontsize='x-small', color=text_color))

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))


def pareto_padding_tricks_wrapper(file_direc, target_video_list, 
        stats_metric='F1', costs_metric='frame-count', compare_cost=True, 
        plot_no_normalization=True, save_direc=None, save_fname=None, fig_title=None, 
        context_type_list=["whole", "region", "inverse", "pixel"], 
        blank_type_list=["whole", "region", "inverse", "pixel"], blank_val_list=[0.01, 0.1, 0.0001, 0.01]):
    
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])

    for ctx_idx, context_type in enumerate(context_type_list):
        for blank_idx, blank_type in enumerate(blank_type_list):
            if ctx_idx == 0 and blank_idx == 0:
                plot_baseline = True
            else:
                plot_baseline = False
            blank_val = blank_val_list[blank_idx]
            single_pareto_trick(file_direc, target_video_list, stats_metric, costs_metric, \
                ax, compare_cost, plot_baseline, plot_baseline, need_method_text=False, 
                context_type=context_type, blank_type=blank_type, blank_val=blank_val_list[blank_idx])
        if plot_no_normalization:
            single_pareto_trick(file_direc, target_video_list, stats_metric, costs_metric, \
                ax, compare_cost, False, need_method_text=False, 
                context_type=context_type, blank_type="whole", blank_val=0.0)

    ax.legend(loc="best")
    if fig_title:
        ax.set_title(fig_title)
    else:
        ax.set_title("Padding methods pareto")
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname))
    else:
        plt.show()
    plt.close()


def get_method_statistics(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count', 
        context_type_idx=0, context_val_idx=1, blank_type_idx=2, blank_val_idx=3, 
        inter_iou_idx=5, merge_iou_idx=7, resize_val_idx=9, filter_type_idx=11, 
        context_type=None, blank_type=None, blank_val=None, 
        merge_iou=None, inter_iou=None, filter_type=None):
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
        context_type=context_type, blank_type=blank_type, blank_val=blank_val,
        merge_iou=merge_iou, inter_iou=inter_iou, filter_type=filter_type)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_list, 
        context_type=context_type, blank_type=blank_type, blank_val=blank_val, 
        merge_iou=merge_iou, inter_iou=inter_iou, filter_type=filter_type)
    
    # For each method: times stat larger than DDS, costs
    method_statistics_dict = {}
    for method_name in costs_result_dict.keys():
        method_statistics_dict[method_name] = [0, 0, []]
        for target_video_name in target_video_list:
            dds_cost = costs_baseline_result_dict['dds'][target_video_name]
            dds_stat = stats_baseline_result_dict['dds'][target_video_name]
            method_cost = costs_result_dict[method_name][target_video_name]
            method_stat = stats_result_dict[method_name][target_video_name]
            if method_cost < dds_cost and method_stat >= dds_stat:
                method_statistics_dict[method_name][0] += 1
                method_statistics_dict[method_name][2].append(method_cost)
            if method_cost < dds_cost:
                method_statistics_dict[method_name][1] += 1
        method_statistics_dict[method_name][2] = np.mean(method_statistics_dict[method_name][2])
    
    sorted_method_tuple_list = sorted(
        method_statistics_dict.items(), key=lambda x:x[1][0], reverse=True)
    for method_tuple in sorted_method_tuple_list:
        print(method_tuple)


def get_method_pareto_count(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count', 
        context_type_idx=0, context_val_idx=1, blank_type_idx=2, blank_val_idx=3, 
        inter_iou_idx=5, merge_iou_idx=7, resize_val_idx=9, filter_type_idx=11, 
        context_type=None, blank_type=None, blank_val=None, 
        merge_iou=None, inter_iou=None, filter_type=None):
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
        context_type=context_type, blank_type=blank_type, blank_val=blank_val,
        merge_iou=merge_iou, inter_iou=inter_iou, filter_type=filter_type)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_list, 
        context_type=context_type, blank_type=blank_type, blank_val=blank_val, 
        merge_iou=merge_iou, inter_iou=inter_iou, filter_type=filter_type)
    
    # For each method: times method appear in each video pareto boundary
    method_statistics_dict = {}
    for target_video_name in target_video_list:
        stat_cost_pair_list = []
        for method_name in costs_result_dict.keys():
            dds_cost = costs_baseline_result_dict['dds'][target_video_name]
            dds_stat = stats_baseline_result_dict['dds'][target_video_name]
            method_cost = costs_result_dict[method_name][target_video_name]
            method_stat = stats_result_dict[method_name][target_video_name]
            if method_cost < dds_cost and method_stat >= dds_stat:
                stat_cost_pair_list.append((method_stat, method_cost, method_name))
        if len(stat_cost_pair_list) == 0:
            continue

        costs_list, stats_list, method_list = pareto_line_utils(
            None, stat_cost_pair_list, dominates_stat_acc, x_idx=1, y_idx=0, \
            choose_metric=choose_metric)  
        for method_name in method_list:
            if method_name not in method_statistics_dict.keys():
                method_statistics_dict[method_name] = 0
            method_statistics_dict[method_name] += 1    
    
    sorted_method_tuple_list = sorted(
        method_statistics_dict.items(), key=lambda x:x[1], reverse=True)
    for method_tuple in sorted_method_tuple_list:
        print(method_tuple)


def scatter_methods(file_direc, target_video_list, 
        stats_metric='F1', costs_metric='frame-count', compare_cost=True,
        save_direc=None, save_fname=None, fig_title=None,
        context_type_idx=0, context_val_idx=1, blank_type_idx=2, blank_val_idx=3, 
        inter_iou_idx=5, merge_iou_idx=7, resize_val_idx=9, filter_type_idx=11, 
        context_type=None, blank_type=None, blank_val=None, 
        merge_iou=None, inter_iou=None, filter_type=None):
    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    stats_list = []
    costs_list = []
    for target_video_name in target_video_list:
        cur_costs_list = read_logs(os.path.join(file_direc, target_video_name, 'costs'))
        costs_list.extend(cur_costs_list)
        cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'stats'))
        stats_list.extend(cur_stats_list)
    
    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=target_video_list, 
        context_type=context_type, blank_type=blank_type, blank_val=blank_val,
        merge_iou=merge_iou, inter_iou=inter_iou, filter_type=filter_type)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_list, 
        context_type=context_type, blank_type=blank_type, blank_val=blank_val, 
        merge_iou=merge_iou, inter_iou=inter_iou, filter_type=filter_type)
    
    # Baseline points
    for method_name, method_result_dict in costs_baseline_result_dict.items():
        baseline_mean_cost = np.mean(list(method_result_dict.values()))
        baseline_mean_stat = np.mean(list(stats_baseline_result_dict[method_name].values()))
        ax.scatter(baseline_mean_cost, baseline_mean_stat, label=method_name)
    dds_mean_cost = np.mean(list(costs_baseline_result_dict['dds'].values()))

    point_stats = []
    point_costs = []
    for method_name in costs_result_dict.keys():
        method_mean_cost = np.mean(list(costs_result_dict[method_name].values()))
        method_mean_stat = np.mean(list(stats_result_dict[method_name].values()))
        if compare_cost and method_mean_cost > dds_mean_cost:
            continue
        if method_mean_cost > dds_mean_cost * 2 / 3:
            print(method_name, method_mean_stat, method_mean_cost)
        point_stats.append(method_mean_stat)
        point_costs.append(method_mean_cost)
    ax.scatter(point_costs, point_stats)

    ax.legend(loc="best")
    if fig_title:
        ax.set_title(fig_title)
    else:
        ax.set_title("Method scatter")
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname))
    else:
        plt.show()
    plt.close()
