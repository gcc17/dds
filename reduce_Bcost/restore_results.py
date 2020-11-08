import os
import ipdb
import cv2 as cv
import numpy as np
import shutil
from .streamB_utils import (find_region_same_id, region_iou)
from dds_utils import (merge_boxes_in_results, Region, Results, visualize_regions, \
    visualize_single_regions)
from collections import Counter


def restore_detection_box(pad_shift_results, pad_regions_list, 
        padded_region_map, proposed_region_map, 
        src_image_w, src_image_h,
        inter_large=True, iou_thresh=0.0):
    """Restore detection box to the location of corresponding regions
        Restore by iou between detection box or padded/proposed region
        Choose the region with the largest IoU or use IoU threshold

    Args:
        pad_shift_results (Results): detection results on padded and shifted regions' image
        pad_regions_list (list): list of padded ShiftRegion
        padded_region_map (dict): key: bin id of packed images; value: padded region_id map in a bin
        proposed_region_map (dict): key: bin id of packed images; value: proposed region_id map in a bin
        inter_large (bool, optional): whether to intersect with the padded region or proposed region
        iou_thresh (float, optional): whether to use iou threshold or region with the largest iou
    """

    restored_pad_shift_results = Results()
    restored_req_regions = []

    # detection result x,y,w,h are relative value
    for single_pad_shift_result in pad_shift_results.regions:
        rx0 = int(single_pad_shift_result.x * src_image_w)
        ry0 = int(single_pad_shift_result.y * src_image_h)
        rx1 = int(single_pad_shift_result.w * src_image_w) + rx0
        ry1 = int(single_pad_shift_result.h * src_image_h) + ry0
        bid = single_pad_shift_result.fid
        if inter_large:
            single_region_map = padded_region_map[bid][ry0:ry1, rx0:rx1]
        else:
            single_region_map = proposed_region_map[bid][ry0:ry1, rx0:rx1]
        single_region_map = single_region_map.reshape(-1)
        cnt = Counter(single_region_map)

        region_id_dict = {}
        most_iou = -1
        for region_id, intersect_pixel in cnt.items():
            if region_id == -1:
                continue
            idx = find_region_same_id(region_id, pad_regions_list)
            if idx is None:
                continue
            if inter_large:
                region_pixel = pad_regions_list[idx].abs_neigh_w * pad_regions_list[idx].abs_neigh_h
            else:
                region_pixel = pad_regions_list[idx].abs_resize_ori_w * pad_regions_list[idx].abs_resize_ori_h
            detect_box_pixel = (rx1-rx0) * (ry1-ry0)
            cur_iou = intersect_pixel / (region_pixel + detect_box_pixel-intersect_pixel)

            # check if iou between detection box and region is larger than threshold or choose the largest
            if iou_thresh == 0:
                if cur_iou > most_iou:
                    most_iou = cur_iou
                    region_id_dict.clear()
                    region_id_dict[region_id] = idx
            elif cur_iou > iou_thresh:
                region_id_dict[region_id] = idx

        if len(region_id_dict) == 0:
            continue

        for most_region_id, idx in region_id_dict.items():
            assert(most_region_id == pad_regions_list[idx].region_id)
            shift_x = pad_regions_list[idx].shift_x
            shift_y = pad_regions_list[idx].shift_y
            fid = pad_regions_list[idx].original_region.fid
            fx = pad_regions_list[idx].fx
            fy = pad_regions_list[idx].fy
            
            result_x = single_pad_shift_result.x
            result_y = single_pad_shift_result.y
            result_w = single_pad_shift_result.w
            result_h = single_pad_shift_result.h
            # the left upper corner of the context region is fixed
            relative_neigh_new_x = shift_x + pad_regions_list[idx].relative_neigh_x
            relative_neigh_new_y = shift_y + pad_regions_list[idx].relative_neigh_y
            # shrink detection box left upper corner relative to the region
            result_x = relative_neigh_new_x + (result_x-relative_neigh_new_x) / fx
            result_y = relative_neigh_new_y + (result_y-relative_neigh_new_y) / fy
            # shrink box width and height
            result_w /= fx
            result_h /= fy

            # restore the detection box x,y,w,h according to shifting
            single_pad_shift_result.x = (0 if result_x<shift_x else result_x-shift_x)
            single_pad_shift_result.y = (0 if result_y<shift_y else result_y-shift_y)
            single_pad_shift_result.w = result_w
            sinlge_pad_shift_result_h = result_h
            if single_pad_shift_result.x + result_w >= 1:
                single_pad_shift_result.w = 1 - single_pad_shift_result.x
            if single_pad_shift_result.y + result_h >= 1:
                single_pad_shift_result.h = 1 - single_pad_shift_result.y
            single_pad_shift_result.fid = fid
            restored_pad_shift_results.append(single_pad_shift_result)
            restored_req_regions.append(pad_regions_list[idx].original_region)

    return restored_pad_shift_results, restored_req_regions


def compress_restored_results(restored_pad_shift_results, min_conf_threshold,
        merge_iou, cover_area_threshold, src_image_w, src_image_h):

    filtered_results = Results()
    for fid, regions_list in restored_pad_shift_results.regions_dict.items():
        region_map = np.zeros((src_image_h, src_image_w), dtype=int)
        regions_list.sort(key=lambda r: r.w*r.h)
        for cur_region in regions_list:
            cur_x0 = int(cur_region.x * src_image_w)
            cur_y0 = int(cur_region.y * src_image_h)
            cur_x1 = int(cur_region.w * src_image_w) + cur_x0
            cur_y1 = int(cur_region.h * src_image_h) + cur_y0
            region_area = (cur_x1-cur_x0) * (cur_y1-cur_y0)

            cur_region_map = region_map[cur_y0:cur_y1, cur_x0:cur_x1]
            covered_area = cur_region_map.sum()
            if covered_area > region_area * cover_area_threshold:
                continue
            region_map[cur_y0:cur_y1, cur_x0:cur_x1] = 1
            filtered_results.append(cur_region)
    
    compressed_results = merge_boxes_in_results(filtered_results.regions_dict, \
        min_conf_threshold, merge_iou)
    return compressed_results


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
        single_region_map = merged_regions_maps[bid][:, :]
        single_region_map = single_region_map.reshape(-1)
        cnt = Counter(single_region_map)

        req_new_regions_list = []
        most_iou = -1
        for merged_region_id in cnt.keys():
            if merged_region_id == -1:
                continue
            for cur_req_new_region_id in merged_new_regions_contain_dict[merged_region_id]:
                cur_req_new_region = req_new_regions_dict[cur_req_new_region_id]
                cur_req_ori_region = cur_req_new_region.original_region
                cur_iou = region_iou(cur_req_new_region, single_pad_shift_result)
                if cur_iou > iou_thresh:         
                    req_new_regions_list.append(cur_req_new_region)
                    # print(cur_iou)
                    # print(bid, cur_req_new_region.x, cur_req_new_region.y, \
                    #     cur_req_new_region.w, cur_req_new_region.h)
                    # print(cur_req_ori_region.fid, cur_req_ori_region.x, cur_req_ori_region.y, \
                    #     cur_req_ori_region.w, cur_req_ori_region.h)

        if len(req_new_regions_list) == 0 or len(req_new_regions_list) > 1:
            continue

        print(bid, single_pad_shift_result.x, single_pad_shift_result.y, 
            single_pad_shift_result.w, single_pad_shift_result.h, round(single_pad_shift_result.conf, 2))
        for cur_req_new_region in req_new_regions_list:
            # get corresponding req_region info
            fx = cur_req_new_region.fx
            fy = cur_req_new_region.fy
            new_x0 = cur_req_new_region.x
            new_y0 = cur_req_new_region.y
            old_x0 = cur_req_new_region.original_region.x
            old_y0 = cur_req_new_region.original_region.y
            fid = cur_req_new_region.original_region.fid
            print(fid, old_x0, old_y0, new_x0, new_y0)

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
            print(fid, result_x, result_y, result_w, result_h, round(single_pad_shift_result.conf, 2))
            restored_req_regions_id.append(cur_req_new_region.region_id)

    return restored_pad_shift_results, restored_req_regions_id
        

def restore_track_regions_detection_box(pad_shift_results, merged_new_regions_dict, 
        track_new_regions_contain_dict, req_new_regions_dict, merged_new_regions_contain_dict,
        merged_regions_maps, src_image_w, src_image_h, src_image_direc=None,
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

        overlap_req_new_regions_dict = {}
        most_iou = -1
        for merged_region_id in cnt.keys():
            if merged_region_id == -1:
                continue
            for cur_req_new_region_id in merged_new_regions_contain_dict[merged_region_id]:
                cur_req_new_region = req_new_regions_dict[cur_req_new_region_id]
                cur_iou = region_iou(cur_req_new_region, single_pad_shift_result)
                if iou_thresh == 0.0:
                    if cur_iou > most_iou:
                        overlap_req_new_regions_dict.clear()
                        overlap_req_new_regions_dict[merged_region_id] = [cur_req_new_region]
                        most_iou = cur_iou
                else:
                    if cur_iou > iou_thresh:         
                        new_x0 = cur_req_new_region.x
                        new_y0 = cur_req_new_region.y
                        if merged_region_id not in overlap_req_new_regions_dict.keys():
                            overlap_req_new_regions_dict[merged_region_id] = []
                        overlap_req_new_regions_dict[merged_region_id].append(cur_req_new_region)

        if len(overlap_req_new_regions_dict) == 0:
            continue
        
        for track_region_id, req_new_regions_list in overlap_req_new_regions_dict.items():

            for cur_req_new_region in req_new_regions_list:
                # get corresponding req_region info
                fx = cur_req_new_region.fx
                fy = cur_req_new_region.fy
                new_x0 = cur_req_new_region.x
                new_y0 = cur_req_new_region.y
                old_x0 = cur_req_new_region.original_region.x
                old_y0 = cur_req_new_region.original_region.y
                fid = cur_req_new_region.original_region.fid

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

            cur_track_merged_region = merged_new_regions_dict[track_region_id]
            cur_track_merged_original_region = cur_track_merged_region.original_region
            cur_ori_x = cur_track_merged_original_region.x
            cur_ori_y = cur_track_merged_original_region.y
            # print(f'tracked region fid {cur_track_merged_original_region.fid}, merged_region_id {track_region_id}, '
            #     f'x {cur_ori_x}, y {cur_ori_y}')
            fx = cur_track_merged_region.fx
            fy = cur_track_merged_region.fy
            
            for (contain_merged_region_id, img_match_loc) in track_new_regions_contain_dict[track_region_id]:
                cur_contain_merged_original_region = merged_new_regions_dict[contain_merged_region_id].original_region
                # print(f'shifted region fid {cur_contain_merged_original_region.fid}, '
                # f'merged_region_id {contain_merged_region_id}, '
                # f'x {cur_contain_merged_original_region.x}, y {cur_contain_merged_original_region.y}')

                box_new_loc_x = single_pad_shift_result.x - cur_track_merged_region.shift_x
                box_new_loc_y = single_pad_shift_result.y - cur_track_merged_region.shift_y
                match_new_loc_x = img_match_loc[0] / src_image_w + cur_ori_x
                match_new_loc_y = img_match_loc[1] / src_image_h + cur_ori_y
                box_new_region = Region(cur_track_merged_original_region.fid, box_new_loc_x, box_new_loc_y, \
                    single_pad_shift_result.w, single_pad_shift_result.h, \
                    1.0, 'object', 1.0)
                match_new_region = Region(-1, match_new_loc_x, match_new_loc_y, \
                    cur_contain_merged_original_region.w, cur_contain_merged_original_region.h, -1, 'object', 1.0)
                
                if src_image_direc:
                    visualize_single_regions(box_new_region, src_image_direc)

                # print(f'box_new_loc: ({box_new_loc_x, box_new_loc_y})')
                # print(f'match_new_loc: ({match_new_loc_x, match_new_loc_y})')
                box_region_iou = region_iou(box_new_region, match_new_region)
                # print(f'box_region_iou: {box_region_iou}')
                if box_region_iou <= 1e-5:
                    continue
                
                track_shift_x = (cur_contain_merged_original_region.x - match_new_loc_x) / fx
                track_shift_y = (cur_contain_merged_original_region.y - match_new_loc_y) / fy
                shift_result_x = cur_track_merged_original_region.x + \
                    (box_new_loc_x-cur_track_merged_original_region.x) / fx
                shift_result_y = cur_track_merged_original_region.y + \
                    (box_new_loc_y-cur_track_merged_original_region.y) / fy

                shift_result_x = max(0, shift_result_x + track_shift_x)
                shift_result_x = min(shift_result_x, 1)
                shift_result_y = max(0, shift_result_y + track_shift_y)
                shift_result_y = min(shift_result_y, 1)
                shift_result_w = min(box_new_region.w / fx, 1-shift_result_x)
                shift_result_h = min(box_new_region.h / fy, 1-shift_result_y)
                shift_fid = cur_contain_merged_original_region.fid
                shift_region = Region(
                    shift_fid, shift_result_x, shift_result_y, shift_result_w, shift_result_h, 
                    single_pad_shift_result.conf, single_pad_shift_result.label, single_pad_shift_result.resolution,
                    single_pad_shift_result.origin
                )

                if src_image_direc:
                    visualize_single_regions(shift_region, src_image_direc, label='tracking')

                restored_pad_shift_results.append(shift_region)
                    
    return restored_pad_shift_results, restored_req_regions_id
        

