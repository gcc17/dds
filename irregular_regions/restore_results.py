import os
import ipdb
import cv2 as cv
import numpy as np
import shutil
from dds_utils import (merge_boxes_in_results, Region, Results, visualize_regions, \
    visualize_single_regions)
from collections import Counter


def restore_detection_boxes(pack_results, all_new_regions_dict, region_area_dict,
        pack_regions_map, src_image_w, src_image_h):

    restored_pack_results = Results()
    restored_region_ids = []

    # detection result x,y,w,h are relative value
    for bid, results_list in pack_results.regions_dict.items():
        for single_result in results_list:
            rx0 = int(single_result.x * src_image_w)
            ry0 = int(single_result.y * src_image_h)
            rx1 = int(single_result.w * src_image_w) + rx0
            ry1 = int(single_result.h * src_image_h) + ry0
            result_area = (rx1 - rx0) * (ry1 - ry0)
            result_overlap_map = pack_regions_map[bid][ry0:ry1, rx0:rx1]
            result_overlap_map = result_overlap_map.reshape(-1)
            cnt = Counter(result_overlap_map)

            most_iou = 0
            most_region_id = 0
            for cur_region_id, overlap_area in cnt.items():
                if cur_region_id == 0:
                    continue
                region_area = region_area_dict[cur_region_id]
                cur_iou = overlap_area / (result_area + region_area - overlap_area)
                if cur_iou > most_iou:
                    most_iou = cur_iou
                    most_region_id = cur_region_id
            if most_iou == 0:
                continue
            most_region = all_new_regions_dict[most_region_id]

            # Restore single_result according to new_region info
            fx = most_region.fx
            fy = most_region.fy
            new_x0 = most_region.x
            new_y0 = most_region.y
            old_x0 = most_region.original_region.x
            old_y0 = most_region.original_region.y
            fid = most_region.original_region.fid
            
            result_x = single_result.x
            result_y = single_result.y
            result_w = single_result.w
            result_h = single_result.h
            
            # Resize
            result_w /= fx
            result_h /= fy
            # Shift
            result_x = new_x0 + (result_x - new_x0) / fx
            result_y = new_y0 + (result_y - new_y0) / fy
            result_x = max(old_x0 + (result_x - new_x0), 0)
            result_x = min(result_x, 1)
            result_y = max(old_y0 + (result_y - new_y0), 0)
            result_y = min(result_y, 1)
            result_w = min(result_w, 1-result_x)
            result_h = min(result_h, 1-result_y)

            restored_pack_results.append(Region(
                fid, result_x, result_y, result_w, result_h, 
                single_result.conf, single_result.label, single_result.resolution,
                single_result.origin
            ))
            restored_region_ids.append(most_region_id)

    return restored_pack_results, restored_region_ids
        
