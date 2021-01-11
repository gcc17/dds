import numpy as np
import os
import matplotlib.pyplot as plt
from dds_utils import (read_results_dict, Results, merge_boxes_in_results)
from .streamB_utils import (draw_region_rectangle, new_evaluate_all)
import pandas as pd
import shutil
import ipdb
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def get_colors(color_cnt):
    cnames = {
        'blue':                 '#0000FF',
        'brown':                '#A52A2A',
        'purple':               '#800080',
        'red':                  '#FF0000',
        'crimson':              '#DC143C',
        'orange':               '#FFA500',
        'green':                '#008000',
        'greenyellow':          '#ADFF2F',
        'blueviolet':           '#8A2BE2',
        'black':                '#000000',
        'burlywood':            '#DEB887',
        'coral':                '#FF7F50',
        'aliceblue':            '#F0F8FF',
        'antiquewhite':         '#FAEBD7',
        'aqua':                 '#00FFFF',
        'aquamarine':           '#7FFFD4',
        'azure':                '#F0FFFF',
        'beige':                '#F5F5DC',
        'bisque':               '#FFE4C4',
        'blanchedalmond':       '#FFEBCD',        
        'cadetblue':            '#5F9EA0',
        'chartreuse':           '#7FFF00',
        'chocolate':            '#D2691E',      
        'cornflowerblue':       '#6495ED',
        'cornsilk':             '#FFF8DC', 
        'cyan':                 '#00FFFF',
        'darkblue':             '#00008B',
        'darkcyan':             '#008B8B',
        'darkgoldenrod':        '#B8860B',
        'darkgray':             '#A9A9A9',
        'darkgreen':            '#006400',
        'darkkhaki':            '#BDB76B',
        'darkmagenta':          '#8B008B',
        'darkolivegreen':       '#556B2F',
        'darkorange':           '#FF8C00',
        'darkorchid':           '#9932CC',
        'darkred':              '#8B0000',
        'darksalmon':           '#E9967A',
        'darkseagreen':         '#8FBC8F',
        'darkslateblue':        '#483D8B',
        'darkslategray':        '#2F4F4F',
        'darkturquoise':        '#00CED1',
        'darkviolet':           '#9400D3',
        'deeppink':             '#FF1493',
        'deepskyblue':          '#00BFFF',
        'dimgray':              '#696969',
        'dodgerblue':           '#1E90FF',
        'firebrick':            '#B22222',
        'floralwhite':          '#FFFAF0',
        'forestgreen':          '#228B22',
        'fuchsia':              '#FF00FF',
        'gainsboro':            '#DCDCDC',
        'ghostwhite':           '#F8F8FF',
        'gold':                 '#FFD700',
        'goldenrod':            '#DAA520',
        'gray':                 '#808080',
        'honeydew':             '#F0FFF0',
        'hotpink':              '#FF69B4',
        'indianred':            '#CD5C5C',
        'indigo':               '#4B0082',
        'ivory':                '#FFFFF0',
        'khaki':                '#F0E68C',
        'lavender':             '#E6E6FA',
        'lavenderblush':        '#FFF0F5',
        'lawngreen':            '#7CFC00',
        'lemonchiffon':         '#FFFACD',
        'lightblue':            '#ADD8E6',
        'lightcoral':           '#F08080',
        'lightcyan':            '#E0FFFF',
        'lightgoldenrodyellow': '#FAFAD2',
        'lightgreen':           '#90EE90',
        'lightgray':            '#D3D3D3',
        'lightpink':            '#FFB6C1',
        'lightsalmon':          '#FFA07A',
        'lightseagreen':        '#20B2AA',
        'lightskyblue':         '#87CEFA',
        'lightslategray':       '#778899',
        'lightsteelblue':       '#B0C4DE',
        'lightyellow':          '#FFFFE0',
        'lime':                 '#00FF00',
        'limegreen':            '#32CD32',
        'linen':                '#FAF0E6',
        'magenta':              '#FF00FF',
        'maroon':               '#800000',
        'mediumaquamarine':     '#66CDAA',
        'mediumblue':           '#0000CD',
        'mediumorchid':         '#BA55D3',
        'mediumpurple':         '#9370DB',
        'mediumseagreen':       '#3CB371',
        'mediumslateblue':      '#7B68EE',
        'mediumspringgreen':    '#00FA9A',
        'mediumturquoise':      '#48D1CC',
        'mediumvioletred':      '#C71585',
        'midnightblue':         '#191970',
        'mintcream':            '#F5FFFA',
        'mistyrose':            '#FFE4E1',
        'moccasin':             '#FFE4B5',
        'navajowhite':          '#FFDEAD',
        'navy':                 '#000080',
        'oldlace':              '#FDF5E6',
        'olive':                '#808000',
        'olivedrab':            '#6B8E23',
        'orangered':            '#FF4500',
        'orchid':               '#DA70D6',
        'palegoldenrod':        '#EEE8AA',
        'palegreen':            '#98FB98',
        'paleturquoise':        '#AFEEEE',
        'palevioletred':        '#DB7093',
        'papayawhip':           '#FFEFD5',
        'peachpuff':            '#FFDAB9',
        'peru':                 '#CD853F',
        'pink':                 '#FFC0CB',
        'plum':                 '#DDA0DD',
        'powderblue':           '#B0E0E6',
        'rosybrown':            '#BC8F8F',
        'royalblue':            '#4169E1',
        'saddlebrown':          '#8B4513',
        'salmon':               '#FA8072',
        'sandybrown':           '#FAA460',
        'seagreen':             '#2E8B57',
        'seashell':             '#FFF5EE',
        'sienna':               '#A0522D',
        'silver':               '#C0C0C0',
        'skyblue':              '#87CEEB',
        'slateblue':            '#6A5ACD',
        'slategray':            '#708090',
        'snow':                 '#FFFAFA',
        'springgreen':          '#00FF7F',
        'steelblue':            '#4682B4',
        'tan':                  '#D2B48C',
        'teal':                 '#008080',
        'thistle':              '#D8BFD8',
        'tomato':               '#FF6347',
        'turquoise':            '#40E0D0',
        'violet':               '#EE82EE',
        'wheat':                '#F5DEB3',
        'white':                '#FFFFFF',
        'whitesmoke':           '#F5F5F5',
        'yellow':               '#FFFF00',
        'yellowgreen':          '#9ACD32'}
    color_list = list(cnames.values())
    return color_list[:color_cnt]


def get_markers(marker_cnt):
    mnames = ['.', 'o', 'v', '^', '<', '>', 's', 'p', '*', \
        'h', '+', 'x', 'D', '|', ',']
    marker_list = mnames[:marker_cnt]
    return marker_list


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    # scatter the ellipse center point
    ax.scatter(mean_x, mean_y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def read_logs(log_path):
    log_list = []
    if log_path:
        with open(log_path, 'r') as f:
            for line in f.readlines():
                log_list.append(line.strip('\n'))
            
    return log_list


def read_stats_costs(stats_path=None, costs_path=None):
    stats_list = read_logs(stats_path)
    costs_list = read_logs(costs_path)
    return stats_list, costs_list


# trafficcam_1_streamBpack_region_0.0_region_0.0_inter_0.0_merge_1.0_resize_no_filter_False-pack-txt
def pick_list_item(target_list, target_metric, fname_len=15, target_video_name=None, 
        context_type=None, context_val=None, blank_type=None, blank_val=None,
        inter_iou=None, merge_iou=None, resize_val=None, filter_type=None,
        context_type_idx=3, context_val_idx=4, blank_type_idx=5, blank_val_idx=6, 
        inter_iou_idx=8, merge_iou_idx=10, resize_val_idx=12, filter_type_idx=14):

    target_head = target_list[0].split(',')
    for idx, head_item in enumerate(target_head):
        if head_item == target_metric:
            target_idx = idx
            break
    
    target_result_dict = {}
    baseline_result_dict = {}
    for target_item in target_list[1:]:
        # read fname and video_name
        split_items = target_item.split(',')
        res_path = split_items[0]
        res_direc = os.path.split(res_path)[0]
        res_fname = os.path.split(res_path)[1]
        cur_video_name = str(os.path.split(res_direc)[1])
        if isinstance(target_video_name, list):
            if cur_video_name not in target_video_name:
                print(cur_video_name, res_fname, target_video_name)
                continue
        elif target_video_name and (cur_video_name != target_video_name):
            print(cur_video_name, res_fname, target_video_name)
            continue
        
        # find target row
        para_list = res_fname.split('_')
        if len(para_list) != fname_len:
            if len(para_list) < 5:
                continue
            if para_list[2] == 'dds':
                if 'dds' not in baseline_result_dict.keys():
                    baseline_result_dict['dds'] = {}
                baseline_result_dict['dds'][cur_video_name] = float(split_items[target_idx])
                continue 
            baseline_name = '_'.join(para_list[2:])
            if baseline_name not in baseline_result_dict.keys():
                baseline_result_dict[baseline_name] = {}
            baseline_result_dict[baseline_name][cur_video_name] = float(split_items[target_idx])
            continue

        # select target item according to fname
        if isinstance(context_type, list):
            if para_list[context_type_idx] not in context_type:
                continue
        elif context_type and (context_type != para_list[context_type_idx]):
            continue
        if isinstance(context_val, list):
            if float(para_list[context_val_idx]) not in context_val:
                continue
        elif context_val != None and (context_val != float(para_list[context_val_idx])):
            continue
        if isinstance(blank_type, list):
            if para_list[blank_type_idx] not in blank_type:
                continue
        elif blank_type and (blank_type != para_list[blank_type_idx]):
            continue
        if isinstance(blank_val, list):
            if float(para_list[blank_val_idx]) not in blank_val:
                continue
        elif blank_val != None and (blank_val != float(para_list[blank_val_idx])):
            continue
        
        if isinstance(inter_iou, list):
            if float(para_list[inter_iou_idx]) not in inter_iou:
                continue
        elif inter_iou != None and (inter_iou != float(para_list[inter_iou_idx])):
            continue
        if isinstance(merge_iou, list):
            if float(para_list[merge_iou_idx]) not in merge_iou:
                continue
        elif merge_iou != None and (merge_iou != float(para_list[merge_iou_idx])):
            continue
        if isinstance(resize_val, list):
            if str(para_list[resize_val_idx]) not in [str(i) for i in resize_val]:
                continue
        elif resize_val != None and ( str(resize_val) != str(para_list[resize_val_idx]) ):
            continue
        if isinstance(filter_type, list):
            if (para_list[filter_type_idx]) not in filter_type:
                continue
        elif filter_type != None and (filter_type != para_list[filter_type_idx]):
            continue

        method_name = '_'.join(para_list[context_type_idx:])
        if method_name not in target_result_dict.keys():
            target_result_dict[method_name] = {}
        target_result_dict[method_name][cur_video_name] = float(split_items[target_idx])

    return target_result_dict, baseline_result_dict


def sort_by_second_list(lista, listb):
    if len(lista) * len(listb) == 0:
        return lista, listb
    zipped_ab = zip(lista, listb)
    sort_zipped = sorted(zipped_ab, key=lambda x: (x[1], x[0]))
    sort_result = zip(*sort_zipped)
    x_axis, y_axis = [list(x) for x in sort_result]
    return x_axis, y_axis


def greater_than_target(target_val, cur_val, choose_metric='max'):
    if choose_metric == 'max':
        return (cur_val >= target_val)
    if choose_metric == 'min':
        return (cur_val <= target_val)


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def find_best_stat_point(stats_result_dict, costs_result_dict, single_video_name, choose_metric):
    best_method_name = None
    best_stat = None
    
    for method_name, method_result_dict in stats_result_dict.items():
        if single_video_name not in method_result_dict.keys():
            return
        cur_stat = method_result_dict[single_video_name]
        if best_stat is None:
            best_stat = cur_stat
            best_method_name = method_name
        else:
            if greater_than_target(best_stat, cur_stat, choose_metric):
                best_stat = cur_stat
                best_method_name = method_name
    best_cost = costs_result_dict[best_method_name][single_video_name]
    return best_cost, best_stat, best_method_name



context_region_val_list = [0.0, 0.3, 0.5, 1.0]
blank_region_val_list = [0.0, 0.3, 0.5]
context_inverse_val_list = [0.0005, 0.0003, 0.0001, 0.001]
blank_inverse_val_list = [0.0005, 0.0001, 0.001]
context_whole_val_list = [0.01, 0.03, 0.05, 0.07]
blank_whole_val_list = [0.01, 0.03, 0.05]
pixel_whole_val_list = [0.0027, 0.01, 0.038, 0.054]


def find_best_val(stats_dict, costs_dict, dds_mean_stat, dds_mean_cost, choose_metric):
    mean_stats_list = []
    mean_costs_list = []
    method_list = list(stats_dict.keys())
    for method_name in method_list:
        method_mean_stat = np.mean(list(stats_dict[method_name].values()))
        method_mean_cost = np.mean(list(costs_dict[method_name].values()))
        mean_stats_list.append(method_mean_stat)
        mean_costs_list.append(method_mean_cost)

    mean_stats_list, mean_costs_list = sort_by_second_list(mean_stats_list, mean_costs_list)
    best_stat_val = None
    best_cost_val = None
    best_method_name = None
    for idx, cur_stat_val in enumerate(mean_stats_list):
        cur_cost_val = mean_costs_list[idx]
        cur_method_name = method_list[idx]
        if best_stat_val is None:
            best_stat_val = cur_stat_val
            best_cost_val = cur_cost_val
            best_method_name = method_list[idx]

        if cur_cost_val < dds_mean_cost:
            if greater_than_target(dds_mean_stat, cur_stat_val, choose_metric):
                return cur_stat_val, cur_cost_val, cur_method_name
            if greater_than_target(best_stat_val, cur_stat_val, choose_metric):
                best_stat_val = cur_stat_val
                best_cost_val = cur_cost_val
                best_method_name = cur_method_name
        else:
            break
    return best_stat_val, best_cost_val, best_method_name


def get_all_videos_data(file_direc, target_video_list, new_stat, dds_as_gt, \
        stats_metric='F1', costs_metric='frame-count'):
    stats_list = []
    costs_list = []
    new_stats_dict = {}
    for target_video_name in target_video_list:
        cur_costs_list = read_logs(os.path.join(file_direc, target_video_name, 'costs'))
        costs_list.extend(cur_costs_list)
        if new_stat:
            new_stats_path = os.path.join(file_direc, target_video_name, 'new_stats')
            if os.path.exists(new_stats_path):
                os.remove(new_stats_path)
            new_evaluate_all(os.path.join(file_direc, target_video_name), video_name=target_video_name, \
                dds_as_gt=dds_as_gt, stats_metric=stats_metric, new_stats_dict=new_stats_dict)
            cur_stats_list = read_logs(new_stats_path)
            stats_list.extend(cur_stats_list)
        else:
            cur_stats_list = read_logs(os.path.join(file_direc, target_video_name, 'stats'))
            stats_list.extend(cur_stats_list)

    stats_result_dict, stats_baseline_result_dict = pick_list_item(stats_list, stats_metric,
        target_video_name=target_video_list)
    costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric,
        target_video_name=target_video_list)

    return stats_result_dict, stats_baseline_result_dict, costs_result_dict, costs_baseline_result_dict


def each_trick_point(file_direc, target_video_list, stats_metric='F1', costs_metric='frame-count', \
        new_stat=False, dds_as_gt=False, 
        context_type_idx=0, context_val_idx=1, blank_type_idx=2, blank_val_idx=3, 
        area_upper_bound_idx=5, resize_type_idx=6, resize_val_idx=7, inter_iou_idx=9, merge_iou_idx=11):

    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'

    stats_result_dict, stats_baseline_result_dict, costs_result_dict, costs_baseline_result_dict = \
        get_all_videos_data(file_direc, target_video_list, new_stat, dds_as_gt, \
            stats_metric, costs_metric)

    # baseline points
    stats_mean_baseline_dict = {}
    costs_mean_baseline_dict = {}
    for cost_key, cost_val_dict in costs_baseline_result_dict.items():
        stat_val_dict = stats_baseline_result_dict[cost_key]
        costs_mean_baseline_dict[cost_key] = np.mean(list(cost_val_dict.values()))
        stats_mean_baseline_dict[cost_key] = np.mean(list(stat_val_dict.values()))
    dds_mean_stat = stats_mean_baseline_dict['dds']
    dds_mean_cost = costs_mean_baseline_dict['dds']
    print(dds_mean_stat, dds_mean_cost)
    print(stats_mean_baseline_dict, costs_mean_baseline_dict)
    
    # no trick points
    no_trick_stats_dict = {}
    no_trick_costs_dict = {}
    ratio1_area_bound = None
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
        merge_iou = float(cur_para_list[merge_iou_idx])
        if merge_iou != 1.0:
            continue
        no_trick_costs_dict[method_name] = method_result_dict
        no_trick_stats_dict[method_name] = stats_result_dict[method_name]
    
    no_trick_stat, no_trick_cost, no_trick_method = find_best_val(
        no_trick_stats_dict, no_trick_costs_dict, \
        dds_mean_stat, dds_mean_cost, choose_metric)
    print(no_trick_stats_dict)
    print(no_trick_stat, no_trick_cost, no_trick_method)
    

    # padding points
    padding_stats_dict = {}
    padding_costs_dict = {}
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
        merge_iou = float(cur_para_list[merge_iou_idx])
        if merge_iou != 1.0:
            continue
        padding_costs_dict[method_name] = method_result_dict
        padding_stats_dict[method_name] = stats_result_dict[method_name]

    padding_stat, padding_cost, padding_method = find_best_val(padding_stats_dict, padding_costs_dict, \
        dds_mean_stat, dds_mean_cost, choose_metric)
    print(padding_stat, padding_cost, padding_method)

    # resize points
    resizing_stats_dict = {}
    resizing_costs_dict = {}
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
        merge_iou = float(cur_para_list[merge_iou_idx])
        if merge_iou != 1.0:
            continue
        resizing_costs_dict[method_name] = method_result_dict
        resizing_stats_dict[method_name] = stats_result_dict[method_name]
    
    resizing_stat, resizing_cost, resizing_method = find_best_val(resizing_stats_dict, resizing_costs_dict, \
        dds_mean_stat, dds_mean_cost, choose_metric)
    print(resizing_stat, resizing_cost, resizing_method)
    
    # merge points
    merge_stats_dict = {}
    merge_costs_dict = {}
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
        merge_iou = float(cur_para_list[merge_iou_idx])
        if merge_iou != 0.0:
            continue
        merge_costs_dict[method_name] = method_result_dict
        merge_stats_dict[method_name] = stats_result_dict[method_name]
    
    merge_stat, merge_cost, merge_method = find_best_val(merge_stats_dict, merge_costs_dict, \
        dds_mean_stat, dds_mean_cost, choose_metric)
    print(merge_stat, merge_cost, merge_method)

    # pad and resize
    pad_resize_stats_dict = {}
    pad_resize_costs_dict = {}
    for method_name, method_result_dict in costs_result_dict.items():
        cur_para_list = (str(method_name)).split('_')
        context_val = float(cur_para_list[context_val_idx])
        blank_val = float(cur_para_list[blank_val_idx])
        if context_val == 0.0 and blank_val == 0.0:
            continue
        resize_type = cur_para_list[resize_type_idx]
        resize_val = float(cur_para_list[resize_val_idx])
        if resize_type == 'ratio' and resize_val == 1:
            continue
        merge_iou = float(cur_para_list[merge_iou_idx])
        if merge_iou != 1.0:
            continue
        pad_resize_costs_dict[method_name] = method_result_dict
        pad_resize_stats_dict[method_name] = stats_result_dict[method_name]

    pad_resize_stat, pad_resize_cost, pad_resize_method = find_best_val(
        pad_resize_stats_dict, pad_resize_costs_dict, \
        dds_mean_stat, dds_mean_cost, choose_metric)
    print(pad_resize_stat, pad_resize_cost, pad_resize_method)

    # pad and merge
    pad_merge_stats_dict = {}
    pad_merge_costs_dict = {}
    for method_name, method_result_dict in costs_result_dict.items():
        cur_para_list = (str(method_name)).split('_')
        context_val = float(cur_para_list[context_val_idx])
        blank_val = float(cur_para_list[blank_val_idx])
        if context_val == 0.0 and blank_val == 0.0:
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
        merge_iou = float(cur_para_list[merge_iou_idx])
        if merge_iou != 0.0:
            continue

        pad_merge_costs_dict[method_name] = method_result_dict
        pad_merge_stats_dict[method_name] = stats_result_dict[method_name]
    
    pad_merge_stat, pad_merge_cost, pad_merge_method = find_best_val(
        pad_merge_stats_dict, pad_merge_costs_dict, \
        dds_mean_stat, dds_mean_cost, choose_metric)
    print(pad_merge_stat, pad_merge_cost, pad_merge_method)

    # resize and merge
    resize_merge_stats_dict = {}
    resize_merge_costs_dict = {}
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
        merge_iou = float(cur_para_list[merge_iou_idx])
        if merge_iou != 0.0:
            continue
        resize_merge_costs_dict[method_name] = method_result_dict
        resize_merge_stats_dict[method_name] = stats_result_dict[method_name]

    resize_merge_stat, resize_merge_cost, resize_merge_method = find_best_val(
        resize_merge_stats_dict, resize_merge_costs_dict, \
        dds_mean_stat, dds_mean_cost, choose_metric)
    print(resize_merge_stat, resize_merge_cost, resize_merge_method)


    # all tricks
    all_tricks_stats_dict = {}
    all_tricks_costs_dict = {}
    for method_name, method_result_dict in costs_result_dict.items():
        cur_para_list = (str(method_name)).split('_')
        context_val = float(cur_para_list[context_val_idx])
        blank_val = float(cur_para_list[blank_val_idx])
        if context_val == 0.0 and blank_val == 0.0:
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
        merge_iou = float(cur_para_list[merge_iou_idx])
        if merge_iou != 0.0:
            continue
        all_tricks_costs_dict[method_name] = method_result_dict
        all_tricks_stats_dict[method_name] = stats_result_dict[method_name]
    
    all_tricks_stat, all_tricks_cost, all_tricks_method = find_best_val(
        all_tricks_stats_dict, all_tricks_costs_dict, dds_mean_stat, dds_mean_cost, choose_metric
    )
    print(all_tricks_stat, all_tricks_cost, all_tricks_method)

    return (stats_mean_baseline_dict, costs_mean_baseline_dict, \
        no_trick_stat, no_trick_cost, no_trick_method, 
        padding_stat, padding_cost, padding_method, resizing_stat, resizing_cost, resizing_method, \
        merge_stat, merge_cost, merge_method,\
        pad_resize_stat, pad_resize_cost, pad_resize_method, pad_merge_stat, pad_merge_cost, pad_merge_method,\
        resize_merge_stat, resize_merge_cost, resize_merge_method, \
        all_tricks_stat, all_tricks_cost, all_tricks_method
    )


def dominates_stat_acc(row_a, row_b, choose_metric):
    # if row_a dominates row_b: row_a has higher accuracy (first element) and smaller cost (second element)
    return greater_than_target(row_b[0], row_a[0], choose_metric) and (row_a[1] <= row_b[1])

def dominates(row, candidateRow, choose_metric=None):
    return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row) 

def simple_cull(inputPoints, dominates, choose_metric=None):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    if len(inputPoints) == 0:
        print("No input points!")
        return
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if choose_metric:
                if dominates(candidateRow, row, choose_metric):
                    # If it is worse on all features remove the row from the array
                    inputPoints.remove(row)
                    dominatedPoints.add(tuple(row))
                elif dominates(row, candidateRow, choose_metric):
                    nonDominated = False
                    dominatedPoints.add(tuple(candidateRow))
                    rowNr += 1
                else:
                    rowNr += 1
            else:
                if dominates(candidateRow, row):
                    # If it is worse on all features remove the row from the array
                    inputPoints.remove(row)
                    dominatedPoints.add(tuple(row))
                elif dominates(row, candidateRow):
                    nonDominated = False
                    dominatedPoints.add(tuple(candidateRow))
                    rowNr += 1
                else:
                    rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            print(candidateRow)
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints


def pareto_line_utils(ax, pair_list, dominate_func, x_idx=0, y_idx=1, choose_metric=None, plot_label=None):
    paretoPoints, dominatedPoints = simple_cull(pair_list, dominate_func, choose_metric)
    x_list = []
    y_list = []
    method_list = []
    for single_pareto_point in paretoPoints:
        x_list.append(single_pareto_point[x_idx])
        y_list.append(single_pareto_point[y_idx])
        if len(single_pareto_point) > 2:
            method_list.append(single_pareto_point[-1])
    
    tmp_x_list = x_list[:]
    y_list, x_list = sort_by_second_list(y_list, x_list)
    method_list, _ = sort_by_second_list(method_list, tmp_x_list)
    if ax:
        if plot_label:
            ax.plot(x_list, y_list, label=plot_label)
        else:
            ax.plot(x_list, y_list)
    return x_list, y_list, method_list


def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认用 box_plot（scale=3）进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        print(iqr, val_low, val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    
    return data_n


def filter_all(data, scale=1):
    all_columns = data.columns
    for idx, col_name in enumerate(all_columns):
        data = outliers_proc(data, col_name, scale)
    return data


def area_box_graph(area_dict, graph_title, save_direc=None, save_fname=None):
    series_dict = {}
    for area_key, area_list in area_dict.items():
        series_dict[area_key] = pd.Series(np.array(area_list))
    df_area = pd.DataFrame(series_dict)
    df_area = filter_all(df_area)
    df_area.boxplot(sym='r*')
    plt.legend(loc='best')
    plt.title(graph_title)
    if save_direc and save_fname:
        os.makedirs(save_direc, exist_ok=True)
        plt.savefig(os.path.join(save_direc, save_fname))
        print(f"save box graph {save_fname} in {save_direc}")
    # plt.show()
    plt.close()        

