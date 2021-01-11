from dds_utils import (read_results_dict, Results)
import numpy as np
from .streamB_utils import (filter_true_regions)

def get_area_percent(area_list, percent_list):
    area_array = np.array(area_list)
    percent_area_list = []
    for cur_percent in percent_list:
        percent_area_list.append(np.percentile(area_array, cur_percent))
    return percent_area_list

# percent_list: [25, 50, 75]
def generate_test_regions(target_region_path, test_region_path, percent_list, region_cnt, video_name, \
        test_confid_thresh=0.5, test_max_area_thresh=0.3):
    target_regions_dict = read_results_dict(target_region_path)
    test_regions_dict = read_results_dict(test_region_path)
    target_area_list = []
    for fid, regions_list in target_regions_dict.items():
        for single_region in regions_list:
            target_area_list.append(single_region.w * single_region.h)
    percent_area_list = get_area_percent(target_area_list, percent_list)
    print(percent_area_list)

    group_region_cnt = int(region_cnt / (len(percent_list) + 1))
    new_regions_result = Results()
    for i in range(len(percent_list) + 1):
        lower_area = 0 if i==0 else percent_area_list[i-1]
        upper_area = 1 if i==len(percent_list) else percent_area_list[i]
        find_region = 0
        for fid, regions_list in test_regions_dict.items():
            regions_list = filter_true_regions(regions_list, test_confid_thresh, test_max_area_thresh)
            if find_region >= group_region_cnt:
                break
            for single_region in regions_list:
                if lower_area <= single_region.w * single_region.h < upper_area:
                    new_regions_result.append(single_region)
                    find_region += 1
                    if find_region >= group_region_cnt:
                        break
    new_regions_result.write(f"{video_name}-test_req_regions")

        


