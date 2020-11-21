from .new_region import (PadShiftRegion)
import os
import ipdb
import cv2 as cv
import numpy as np
import shutil
from math import sqrt
from .streamB_utils import (normalize_image, merge_one_frame_regions)
from dds_utils import (Results)
import copy
from collections import OrderedDict


def pad_single_req_region_context(original_region, region_id, context_whole_w, context_whole_h):
    new_x0 = max(0, original_region.x - context_whole_w)
    new_y0 = max(0, original_region.y - context_whole_h)
    new_x1 = min(1, new_x0 + original_region.w + 2*context_whole_w)
    new_y1 = min(1, new_y0 + original_region.h + 2*context_whole_h)
    new_w = new_x1 - new_x0
    new_h = new_y1 - new_y0
    return PadShiftRegion(original_region, region_id, new_x0, new_y0, new_w, new_h)


def judge_padding_value(original_region_item, padding_type, padding_val):
    padding_val = float(padding_val)
    if padding_type == 'whole':
        return padding_val, padding_val
    if padding_type == 'region':
        return padding_val*original_region_item.w, padding_val*original_region_item.h
    if padding_type == 'inverse':
        return padding_val/original_region_item.w, padding_val/original_region_item.h
    if padding_type == 'pixel':
        args = [4, 2*(original_region_item.w + original_region_item.h), -padding_val]
        padding_side_roots = np.roots(args)
        padding_side = np.max(padding_side_roots)
        return padding_side, padding_side


def pad_filtered_regions_context(
        req_regions_result, context_padding_type, context_val, merge_iou=0.0, 
        start_req_region_id=0, start_merged_region_id=0
    ):

    # Pad filtered req_regions with context
    cur_req_region_id = start_req_region_id
    cur_merged_region_id = start_merged_region_id
    req_new_regions_dict = {}
    merged_new_regions_dict = OrderedDict()
    merged_new_regions_contain_dict = {}

    # Enumerate req_regions in each frame
    for fid, original_regions_list in req_regions_result.regions_dict.items():
        new_regions_list = []
        # Pad the original req_regions with context
        for original_region_item in original_regions_list:
            context_whole_w, context_whole_h = \
                judge_padding_value(original_region_item, context_padding_type, context_val)
            new_region_item = pad_single_req_region_context(
                original_region_item, cur_req_region_id, context_whole_w, context_whole_h)
            new_regions_list.append(new_region_item)
            req_new_regions_dict[cur_req_region_id] = new_region_item
            cur_req_region_id += 1
        # Merge small new regions into large new region
        merged_new_regions_list, frame_merged_region_contain_dict = \
            merge_one_frame_regions(new_regions_list, cur_merged_region_id, merge_iou)
        
        # Update merged regions dict and contain_dict
        for merged_new_region in merged_new_regions_list:
            merged_new_regions_dict[merged_new_region.region_id] = merged_new_region
        merged_new_regions_contain_dict.update(frame_merged_region_contain_dict)
        cur_merged_region_id += len(merged_new_regions_list)

    # Recover req_regions xywh, because we no longer need context-padded req_region
    for req_region_id in req_new_regions_dict.keys():
        req_ori_region = req_new_regions_dict[req_region_id].original_region
        req_new_regions_dict[req_region_id].x = req_ori_region.x
        req_new_regions_dict[req_region_id].y = req_ori_region.y
        req_new_regions_dict[req_region_id].w = req_ori_region.w
        req_new_regions_dict[req_region_id].h = req_ori_region.h

    return req_new_regions_dict, merged_new_regions_dict, merged_new_regions_contain_dict


def pad_filtered_small_regions_context(req_regions_result, context_padding_type, context_val, 
        merge_iou=0.0, start_req_region_id=0, start_merged_region_id=0, 
        resize_ratio_list=[2, 3]):
    
    # Pad filtered req_regions with context
    cur_req_region_id = start_req_region_id
    tmp_cur_req_region_id = start_req_region_id
    cur_merged_region_id = start_merged_region_id
    tmp_cur_merged_region_id = start_merged_region_id
    tmp_req_new_regions_dict = {}
    req_new_regions_dict = {}
    merged_new_regions_dict = OrderedDict()
    merged_new_regions_contain_dict = {}
    # Resize merged_new_region with this ratio
    # Then change contained req_regions
    # In blank padding: do resizing

    # Enumerate req_regions in each frame
    for fid, original_regions_list in req_regions_result.regions_dict.items():
        new_regions_list = []
        # Pad the original req_regions with context
        for original_region_item in original_regions_list:
            context_whole_w, context_whole_h = \
                judge_padding_value(original_region_item, context_padding_type, context_val)
            new_region_item = pad_single_req_region_context(
                original_region_item, tmp_cur_req_region_id, context_whole_w, context_whole_h)
            new_regions_list.append(new_region_item)
            tmp_req_new_regions_dict[tmp_cur_req_region_id] = new_region_item
            tmp_cur_req_region_id += 1
        # Merge small new regions into large new region
        merged_new_regions_list, frame_merged_region_contain_dict = \
            merge_one_frame_regions(new_regions_list, tmp_cur_merged_region_id, merge_iou)
        tmp_cur_merged_region_id += len(merged_new_regions_list)

        for merged_new_region in merged_new_regions_list:
            for resize_ratio in resize_ratio_list:
                merged_new_region_id = merged_new_region.region_id
                fx = min(resize_ratio, 1/merged_new_region.w)
                fy = min(resize_ratio, 1/merged_new_region.h, fx)
                resize_merged_new_region = PadShiftRegion(
                    merged_new_region.original_region, cur_merged_region_id, 
                    merged_new_region.x, merged_new_region.y, merged_new_region.w*fx, merged_new_region.h*fy, 
                    fx=fx, fy=fy
                )

                # Update merged regions dict
                merged_new_regions_dict[cur_merged_region_id] = resize_merged_new_region
                merged_new_regions_contain_dict[cur_merged_region_id] = []

                # change small new regions xywh: original xywh after resizing
                # because we only consider req_regions' intersection with detection box
                # we no longer need context-padded regions' xywh
                # copy the tmp_req_region and change its xywh, fx, fy
                for contain_region_id in frame_merged_region_contain_dict[merged_new_region_id]:
                    cur_new_region = tmp_req_new_regions_dict[contain_region_id]
                    cur_ori_region = cur_new_region.original_region

                    # change req_new_region xywh
                    new_x = merged_new_region.x + (cur_ori_region.x - merged_new_region.x) * fx
                    new_y = merged_new_region.y + (cur_ori_region.y - merged_new_region.y) * fy
                    new_w = cur_ori_region.w * fx
                    new_h = cur_ori_region.h * fy
                    resize_req_new_region = PadShiftRegion(
                        cur_ori_region, cur_req_region_id, new_x, new_y, new_w, new_h, 
                        fx=fx, fy=fy
                    )
                    req_new_regions_dict[cur_req_region_id] = resize_req_new_region
                    merged_new_regions_contain_dict[cur_merged_region_id].append(cur_req_region_id)
                    cur_req_region_id += 1
                
                cur_merged_region_id += 1
    
    return req_new_regions_dict, merged_new_regions_dict, merged_new_regions_contain_dict


def save_padded_region(
        src_image, merged_new_region, blank_padding_type, blank_val, padded_regions_direc, 
        src_image_w, src_image_h
    ):

    # Crop out the blank-padded region and save it
    blank_whole_w, blank_whole_h = judge_padding_value(merged_new_region.original_region, \
        blank_padding_type, blank_val)
    new_w = min(1, merged_new_region.w + 2*blank_whole_w)
    new_h = min(1, merged_new_region.h + 2*blank_whole_h)
    abs_delta_x = int( (new_w - merged_new_region.w) / 2 * src_image_w )
    abs_delta_y = int( (new_h - merged_new_region.h) / 2 * src_image_h )
    abs_new_w = int(src_image_w * new_w)
    abs_new_h = int(src_image_h * new_h)

    # Set the blank-padded region background as normalization value
    blank_padded_region = np.zeros((abs_new_h, abs_new_w, 3), dtype=np.uint8)
    blank_padded_region = normalize_image(blank_padded_region)
    
    abs_context_x = int(merged_new_region.original_region.x * src_image_w)
    abs_context_y = int(merged_new_region.original_region.y * src_image_h)
    abs_ori_context_w = int(merged_new_region.original_region.w * src_image_w)
    abs_ori_context_h = int(merged_new_region.original_region.h * src_image_h)
    abs_context_w = int(merged_new_region.w * src_image_w)
    abs_context_h = int(merged_new_region.h * src_image_h)

    # Read context-padded region and resize it
    context_padded_region = src_image[abs_context_y:abs_context_y+abs_ori_context_h, \
        abs_context_x:abs_context_x+abs_ori_context_w, :]
    context_padded_region = cv.resize(context_padded_region, (abs_context_w, abs_context_h), 
                                    interpolation=cv.INTER_CUBIC)
    
    # Put the context-padded region on the 'blank' region
    blank_padded_region[abs_delta_y:abs_delta_y+abs_context_h, abs_delta_x:abs_delta_x+abs_context_w, :] = \
        context_padded_region
    # Save region content
    blank_padded_region = cv.cvtColor(blank_padded_region, cv.COLOR_RGB2BGR)
    blank_padded_region_path = os.path.join(
        padded_regions_direc, f'region-{merged_new_region.region_id}.png'
    )
    cv.imwrite(blank_padded_region_path, blank_padded_region, [cv.IMWRITE_PNG_COMPRESSION, 0])

    # Change merged_new_region w,h after blank padding: shifting needs whole size
    merged_new_region.blank_x = merged_new_region.original_region.x - (new_w - merged_new_region.w) / 2
    merged_new_region.blank_y = merged_new_region.original_region.y - (new_h - merged_new_region.h) / 2
    merged_new_region.w = new_w
    merged_new_region.h = new_h

    return merged_new_region, abs_context_w / abs_ori_context_w, abs_context_h / abs_ori_context_h


def pad_filtered_regions_blank(
        req_new_regions_dict, merged_new_regions_dict, merged_new_regions_contain_dict, 
        batch_images_direc, merged_images_direc, blank_padding_type, blank_val, 
        sort_context_region=False
    ):

    padded_regions_direc = f'{batch_images_direc}-padded_regions'
    os.makedirs(padded_regions_direc, exist_ok=True)
    last_image = None
    last_fid = -1
    src_image_w = None
    src_image_h = None

    sorted_merged_new_regions = []
    if sort_context_region:
        for merged_new_region_id, merged_new_region in merged_new_regions_dict.items():
            insert_loc = 0
            for sorted_region_id in sorted_merged_new_regions:
                sorted_new_region = merged_new_regions_dict[sorted_region_id]
                if merged_new_region.original_region.fid < sorted_new_region.original_region.fid:
                    break
                if merged_new_region.original_region.fid > sorted_new_region.original_region.fid:
                    insert_loc += 1
                    continue
                if merged_new_region.x < sorted_new_region.x:
                    break
                if merged_new_region.x > sorted_new_region.x:
                    insert_loc += 1
                    continue
                if merged_new_region.y < sorted_new_region.y:
                    break
                if merged_new_region.y > sorted_new_region.y:
                    insert_loc += 1
                    continue
                if merged_new_region.w < sorted_new_region.w:
                    break
                if merged_new_region.w > sorted_new_region.w:
                    insert_loc += 1
                    continue
                if merged_new_region.h < sorted_new_region.h:
                    break
                if merged_new_region.h > sorted_new_region.h:
                    insert_loc += 1
                    continue
            sorted_merged_new_regions.insert(insert_loc, merged_new_region_id)
        sorted_merged_new_regions_dict = OrderedDict()
        for merged_region_id in sorted_merged_new_regions:
            merged_new_region = merged_new_regions_dict[merged_region_id]
            sorted_merged_new_regions_dict[merged_region_id] = merged_new_region
        merged_new_regions_dict = sorted_merged_new_regions_dict

    for merged_new_region_id in merged_new_regions_dict.keys():
        merged_new_region = merged_new_regions_dict[merged_new_region_id]
        if last_fid == merged_new_region.original_region.fid:
            cur_image = last_image
        else:
            cur_fname = f'{str(merged_new_region.original_region.fid).zfill(10)}.png'
            cur_image_path = os.path.join(merged_images_direc, cur_fname)
            cur_image = cv.imread(cur_image_path)
            cur_image = cv.cvtColor(cur_image, cv.COLOR_BGR2RGB)
            last_image = cur_image
            last_fid = merged_new_region.original_region.fid
        if src_image_w is None:
            src_image_w = cur_image.shape[1]
            src_image_h = cur_image.shape[0]
        if not (src_image_w == cur_image.shape[1] and src_image_h == cur_image.shape[0]):
            print('Image shape does not match!')
            exit()
        
        merged_new_regions_dict[merged_new_region_id], fx, fy = save_padded_region(
            cur_image, merged_new_region, blank_padding_type, blank_val, padded_regions_direc, 
            src_image_w, src_image_h
        )
        for contain_id in merged_new_regions_contain_dict[merged_new_region_id]:
            req_new_regions_dict[contain_id].fx = fx
            req_new_regions_dict[contain_id].fy = fy
        

    return padded_regions_direc, src_image_w, src_image_h, merged_new_regions_dict, req_new_regions_dict
