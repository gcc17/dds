import os
import ipdb
import cv2 as cv
import numpy as np
import shutil
from .streamB_utils import (find_region_same_id, region_iou)
from dds_utils import (merge_boxes_in_results, Region, Results, visualize_regions, \
    visualize_single_regions)
from collections import Counter


def restore_merged_regions_detection_box(pad_shift_results, merged_new_regions_dict, 
        req_new_regions_dict, merged_new_regions_contain_dict, 
        merged_regions_maps, src_image_w, src_image_h,
        iou_thresh=0.2):
    """Restore detection box to the location of corresponding regions
        Restore by iou between detection box and the proposed region
        Choose the region with the largest IoU or use IoU threshold
    """

    restored_pad_shift_results = Results()
    restored_req_regions_id = []

    # detection result x,y,w,h are relative value
    for single_pad_shift_result in pad_shift_results.regions:
        rx0 = int(single_pad_shift_result.x * src_image_w)
        ry0 = int(single_pad_shift_result.y * src_image_h)
        rx1 = int(single_pad_shift_result.w * src_image_w) + rx0
        ry1 = int(single_pad_shift_result.h * src_image_h) + ry0
        bid = single_pad_shift_result.fid
        single_region_map = merged_regions_maps[bid][ry0:ry1, rx0:rx1]
        single_region_map = single_region_map.reshape(-1)
        cnt = Counter(single_region_map)
        # print(bid, rx0, ry0, rx1, ry1, round(single_pad_shift_result.conf, 2))

        req_new_regions_list = []
        most_iou = -1
        for merged_region_id in cnt.keys():
            if merged_region_id == -1:
                continue
            for cur_req_new_region_id in merged_new_regions_contain_dict[merged_region_id]:
                cur_req_new_region = req_new_regions_dict[cur_req_new_region_id]
                cur_iou = region_iou(cur_req_new_region, single_pad_shift_result)
                if cur_iou > iou_thresh:         
                    req_new_regions_list.append(cur_req_new_region)
                # if cur_iou > iou_thresh and cur_iou > most_iou:
                #     most_iou = cur_iou
                #     req_new_regions_list = [cur_req_new_region]

        if len(req_new_regions_list) == 0:
            continue
        
        for cur_req_new_region in req_new_regions_list:
            # get corresponding req_region info
            fx = cur_req_new_region.fx
            fy = cur_req_new_region.fy
            new_x0 = cur_req_new_region.x
            new_y0 = cur_req_new_region.y
            old_x0 = cur_req_new_region.original_region.x
            old_y0 = cur_req_new_region.original_region.y
            fid = cur_req_new_region.original_region.fid
            # print(new_x0, new_y0, cur_req_new_region.w, cur_req_new_region.h)
            # print(old_x0, old_y0, cur_req_new_region.original_region.w, cur_req_new_region.original_region.h)

            # for every possible region, read original single_pad_shift_result info
            result_x = single_pad_shift_result.x
            result_y = single_pad_shift_result.y
            result_w = single_pad_shift_result.w
            result_h = single_pad_shift_result.h
            
            result_w /= fx
            result_h /= fy

            # resize
            result_x = new_x0 + (result_x - new_x0) / fx
            result_y = new_y0 + (result_y - new_y0) / fy
            # shift
            result_x = max(old_x0 + (result_x - new_x0), 0)
            result_x = min(result_x, 1)
            result_y = max(old_y0 + (result_y - new_y0), 0)
            result_y = min(result_y, 1)
            result_w = min(result_w, 1-result_x)
            result_h = min(result_h, 1-result_y)

            restored_pad_shift_results.append(Region(
                fid, result_x, result_y, result_w, result_h, 
                single_pad_shift_result.conf, single_pad_shift_result.label, single_pad_shift_result.resolution,
                single_pad_shift_result.origin
            ))
            restored_req_regions_id.append(cur_req_new_region.region_id)
    return restored_pad_shift_results, restored_req_regions_id
        
