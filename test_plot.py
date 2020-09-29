# import matplotlib.pyplot as plt
# import numpy as np

# x = np.arange(1,100,1)

# fig = plt.figure()

# ax1 = fig.add_subplot(231)
# ax1.plot(x,x)

# ax2 = fig.add_subplot(534)
# ax2.plot(x,x*x)

# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# x = np.arange(1,100,1)

# plt.subplot(231)
# plt.plot(x, x)

# plt.subplot(534)
# plt.plot(x, x*x)

# plt.show()

import os
from reduce_Bcost.plot_functions import (compare_all_acc_cost, compare_padding_method, \
    each_padding_compare, plot_trick_point, plot_pareto_line)
from reduce_Bcost.streamB_utils import (draw_region_rectangle, new_evaluate_all, \
    filter_bad_parameters_wrapper)
from reduce_Bcost.plot_utils import (each_trick_point)

server_result_direc = os.path.join('..', 'server_results_0928')
all_target_videos = ['trafficcam_1']

# save_direc = os.path.join(server_result_direc, 'overall_compare')
# for target_video_name in all_target_videos:
#     stats_path = os.path.join(server_result_direc, target_video_name, 'stats')
#     costs_path = os.path.join(server_result_direc, target_video_name, 'costs')
#     compare_all_acc_cost(stats_path, costs_path, video_names=[target_video_name], \
#         save_direc=save_direc, save_fname=f'{target_video_name}-all.png', compare_mpeg=False)

# save_direc = os.path.join(server_result_direc,'padding_compare')
# for target_video_name in all_target_videos:
#     stats_path = os.path.join(server_result_direc, target_video_name, 'stats')
#     costs_path = os.path.join(server_result_direc, target_video_name, 'costs')
#     compare_padding_method(stats_path, costs_path, video_names=[target_video_name], \
#         save_direc=save_direc, save_fname=f'{target_video_name}-padding.png')

# save_direc = os.path.join(server_result_direc,'each_padding')
# context_type_list = ['region', 'whole', 'inverse', 'pixel']
# blank_type_list = ['region', 'whole', 'inverse']
# for target_video_name in all_target_videos:
#     stats_path = os.path.join(server_result_direc, target_video_name, 'stats')
#     costs_path = os.path.join(server_result_direc, target_video_name, 'costs')
#     for context_type in context_type_list:
#         for blank_type in blank_type_list:
#             each_padding_compare(stats_path, costs_path, target_video_name=target_video_name, \
#                 save_direc=save_direc, save_fname=f'{context_type}-{blank_type}-{target_video_name}.png', 
#                 context_type=context_type, blank_type=blank_type,)


# new_stats_dict = {}
# src_video_direc = os.path.join('..', 'videos')
# for target_video_name in all_target_videos:
#     file_direc = os.path.join(server_result_direc, target_video_name)
#     src_image_direc = os.path.join(src_video_direc, target_video_name, 'src')
#     save_image_direc = os.path.join(server_result_direc, target_video_name, 'vis_evaluate_gt')
#     new_evaluate_all(file_direc, target_video_name, new_stats_dict=new_stats_dict, \
#         src_image_direc=src_image_direc, save_image_direc=save_image_direc, stats_metric='F1', vis_gt=False)

# print(new_stats_dict)

# filter_bad_parameters_wrapper(server_result_direc, all_target_videos, 'frame-count', 'F1')
# each_trick_point(server_result_direc, all_target_videos, new_stat=True)
save_direc = os.path.join(server_result_direc, 'trick_point')
# plot_trick_point(server_result_direc, all_target_videos, new_stat=False, dds_as_gt=False, save_direc=save_direc)
# plot_trick_point(server_result_direc, all_target_videos, new_stat=True, dds_as_gt=False, save_direc=save_direc)
# plot_trick_point(server_result_direc, all_target_videos, new_stat=True, dds_as_gt=True, save_direc=save_direc)
save_direc = os.path.join(server_result_direc, 'pareto_lines')
plot_pareto_line(server_result_direc, all_target_videos, save_direc=save_direc, \
    save_fname='pareto_gt_cost.png', compare_cost=True)
plot_pareto_line(server_result_direc, all_target_videos, save_direc=save_direc, \
    save_fname='pareto_gt.png', compare_cost=False)
plot_pareto_line(server_result_direc, all_target_videos, dds_as_gt=True, \
    save_direc=save_direc, save_fname='pareto_dds_cost.png', compare_cost=True)
plot_pareto_line(server_result_direc, all_target_videos, dds_as_gt=True, \
    save_direc=save_direc, save_fname='pareto_dds.png', compare_cost=False)

