from .new_region import (ShiftRegion, PadShiftRegion)
import os
import ipdb
import cv2 as cv
import numpy as np
import shutil
from math import sqrt
from .streamB_utils import (normalize_image, merge_one_frame_regions)
from dds_utils import (Results)
import copy


def pad_single_req_region_context(original_region, region_id, context_whole_w, context_whole_h):
    new_x0 = max(0, original_region.x - context_whole_w)
    new_y0 = max(0, original_region.y - context_whole_h)
    new_x1 = min(1, new_x0 + original_region.w + 2*context_whole_w)
    new_y1 = min(1, new_y0 + original_region.h + 2*context_whole_h)
    new_w = new_x1 - new_x0
    new_h = new_y1 - new_y0
    return PadShiftRegion(original_region, region_id, new_x0, new_y0, new_w, new_h)


def pad_single_merged_region(merged_new_region, blank_whole_w, blank_whole_h, \
        merged_region_contain_list, req_new_regions_dict,
        region_image, context_image, tmp_regions_direc, \
        resize_type=None, area_upper_bound=None, fixed_ratio=None, fixed_area=None):

    image_w = region_image.shape[1]
    image_h = region_image.shape[0]
    
    # add context(low-quality) to normalization area
    abs_context_x = int(merged_new_region.original_region.x * image_w)
    abs_context_y = int(merged_new_region.original_region.y * image_h)
    abs_context_x1 = int((merged_new_region.original_region.x + merged_new_region.original_region.w) * image_w)
    abs_context_y1 = int((merged_new_region.original_region.y + merged_new_region.original_region.h) * image_h)
    # ensure context-padded region not exceed the original frame size
    abs_context_x = max(0, abs_context_x)
    abs_context_y = max(0, abs_context_y)
    abs_context_x1 = min(abs_context_x1, image_w)
    abs_context_y1 = min(abs_context_y1, image_h)

    abs_context_w = abs_context_x1 - abs_context_x
    abs_context_h = abs_context_y1 - abs_context_y
    # ensure the blank-padded region not exceed the original frame size
    abs_blank_w = min( int(blank_whole_w * image_w), (image_w-abs_context_w) // 2 )
    abs_blank_h = min( int(blank_whole_h * image_h), (image_h-abs_context_h) // 2 )
    abs_total_w = abs_context_w + 2*abs_blank_w
    abs_total_h = abs_context_h + 2*abs_blank_h
    # print(f'abs_context_w: {abs_context_w}, abs_context_h: {abs_context_h}')
    # print(f'abs_total_w: {abs_total_w}, abs_total_h: {abs_total_h}')

    # set the padded region background as normalization value
    region_content = np.zeros((abs_total_h, abs_total_w, 3), dtype=np.uint8)
    region_content = normalize_image(region_content)
    # add the context (low-quality) part
    region_content[abs_blank_h:abs_blank_h+abs_context_h, abs_blank_w:abs_blank_w+abs_context_w, :] = \
        context_image[abs_context_y:abs_context_y+abs_context_h, abs_context_x:abs_context_x+abs_context_w, :]
    
    # add region(high-quality) one by one (contained by the merged region)
    for contain_region_id in merged_region_contain_list:
        cur_new_region = req_new_regions_dict[contain_region_id]
        abs_region_x = max(int(cur_new_region.original_region.x * image_w), abs_context_x)
        abs_region_y = max(int(cur_new_region.original_region.y * image_h), abs_context_y)
        abs_region_x1 = min(int((cur_new_region.original_region.x + cur_new_region.original_region.w)*image_w),\
            abs_context_x1)
        abs_region_y1 = min(int((cur_new_region.original_region.y + cur_new_region.original_region.h)*image_h),\
            abs_context_y1)

        abs_new_x0 = abs_blank_w + abs_region_x - abs_context_x
        abs_new_y0 = abs_blank_h + abs_region_y - abs_context_y
        abs_new_x1 = abs_new_x0 + (abs_region_x1 - abs_region_x)
        abs_new_y1 = abs_new_y0 + (abs_region_y1 - abs_region_y)
        
        # print(region_content[abs_new_y0:abs_new_y1, abs_new_x0:abs_new_x1, :].shape)
        # print(region_image[abs_region_y:abs_region_y1, abs_region_x:abs_region_x1, :].shape)
        # print(region_content.shape)
        # import ipdb; ipdb.set_trace()
        region_content[abs_new_y0:abs_new_y1, abs_new_x0:abs_new_x1, :] = \
            region_image[abs_region_y:abs_region_y1, abs_region_x:abs_region_x1, :]

    # for packing, we use the whole padded region(including blank part)
    # we set xywh as the whole region
    new_x = merged_new_region.original_region.x - blank_whole_w
    new_y = merged_new_region.original_region.y - blank_whole_h
    new_w = min(merged_new_region.original_region.w + 2*blank_whole_w, 1)
    new_h = min(merged_new_region.original_region.h + 2*blank_whole_h, 1)
    
    # resize region if needed
    fx = fy = 1
    if resize_type and area_upper_bound is not None:
        if new_w * new_h <= area_upper_bound:
            if resize_type == 'fixed_ratio' and fixed_ratio:
                area_ratio = fixed_ratio
            elif resize_type == 'fixed_area' and fixed_area:
                area_ratio = fixed_area / (new_w * new_h)
            fx = fy = sqrt(area_ratio)
            fx = min(fx, 1/new_w)
            fy = min(fy, 1/new_h)
            abs_total_w = min(int(fx * abs_total_w), image_w)
            abs_total_h = min(int(fy * abs_total_h), image_h)
            region_content = cv.resize(region_content, (abs_total_w, abs_total_h), \
                    interpolation=cv.INTER_CUBIC)
    new_w = (abs_total_w + 0.8) / image_w
    new_h = (abs_total_h + 0.8) / image_h
    # print(f'w*src {int(new_w * image_w)}, abs_total_w: {abs_total_w}')
    # print(f'h*src {int(new_h * image_h)}, abs_total_h: {abs_total_h}')
            
    # change small new regions xywh: original xywh after resizing
    # because we only consider req_regions' intersection with detection box
    # we no longer need context-padded regions' xywh
    for contain_region_id in merged_region_contain_list:
        cur_new_region = req_new_regions_dict[contain_region_id]
        cur_ori_region = cur_new_region.original_region
        merged_ori_region = merged_new_region.original_region

        # change req_new_region xywh
        cur_new_region.x = merged_ori_region.x + (cur_ori_region.x - merged_ori_region.x) * fx
        cur_new_region.y = merged_ori_region.y + (cur_ori_region.y - merged_ori_region.y) * fy
        cur_new_region.w = cur_ori_region.w * fx
        cur_new_region.h = cur_ori_region.h * fy
        cur_new_region.fx = fx
        cur_new_region.fy = fy

        req_new_regions_dict[contain_region_id] = cur_new_region
    
    # save region content
    region_content = cv.cvtColor(region_content, cv.COLOR_RGB2BGR)
    merged_region_id = merged_new_region.region_id
    full_pad_image_path = os.path.join(tmp_regions_direc, f"region-{merged_region_id}.png")
    cv.imwrite(full_pad_image_path, region_content, [cv.IMWRITE_PNG_COMPRESSION, 0])

    # change merge new region after blank padding and resizing new xywh
    merged_new_region.x = new_x
    merged_new_region.y = new_y
    merged_new_region.w = new_w
    merged_new_region.h = new_h
    merged_new_region.fx = fx
    merged_new_region.fy = fy

    abs_merged_w = int(merged_new_region.w * image_w)
    abs_merged_h = int(merged_new_region.h * image_h)
    if abs_merged_h * abs_merged_w == 0:
        return None
    return merged_new_region


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


def pad_all_req_regions(req_regions_result, \
        region_image_direc, context_image_direc, video_result_name,\
        context_padding_type, context_val, blank_padding_type, blank_val, \
        resize_type=None, area_upper_bound=None, fixed_ratio=None, fixed_area=None, \
        merge_iou=0.0, req_region_id_start=0, merged_region_id_start=0):
    
    cur_req_region_id = req_region_id_start
    cur_merged_region_id = merged_region_id_start
    req_new_regions_dict = {}
    merged_new_regions_dict = {}
    merged_new_regions_contain_dict = {}
    tmp_regions_direc = f'{video_result_name}-tmp_regions'
    os.makedirs(tmp_regions_direc, exist_ok=True)
    for fname in os.listdir(tmp_regions_direc):
        if "png" not in fname:
            continue
        else:
            os.remove(os.path.join(tmp_regions_direc, fname))
    src_image_w = None
    src_image_h = None

    # enumerate req_regions in each frame
    for fid, original_regions_list in req_regions_result.regions_dict.items():
        new_regions_list = []
        # pad the original req_regions with 
        for original_region_item in original_regions_list:
            context_whole_w, context_whole_h = \
                judge_padding_value(original_region_item, context_padding_type, context_val)
            new_region_item = pad_single_req_region_context(original_region_item, cur_req_region_id, \
                context_whole_w, context_whole_h)
            new_regions_list.append(new_region_item)
            req_new_regions_dict[cur_req_region_id] = new_region_item
            cur_req_region_id += 1
        # merge small new regions into large new region
        merged_new_regions_list, frame_merged_region_contain_dict = \
            merge_one_frame_regions(new_regions_list, cur_merged_region_id, merge_iou)

        # read context and region image
        fname = f'{str(fid).zfill(10)}.png'
        region_image_path = os.path.join(region_image_direc, fname)
        print(region_image_path)
        region_image = cv.imread(region_image_path)
        region_image = cv.cvtColor(region_image, cv.COLOR_BGR2RGB)
        context_image_path = os.path.join(context_image_direc, fname)
        context_image = cv.imread(context_image_path)
        context_image = cv.cvtColor(context_image, cv.COLOR_BGR2RGB)
        
        if not src_image_w:
            src_image_w = region_image.shape[1]
            src_image_h = region_image.shape[0]
        assert(src_image_w == region_image.shape[1] and src_image_h == region_image.shape[0])
        context_image = cv.resize(context_image, (src_image_w, src_image_h), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
    
        for idx, merged_new_region in enumerate(merged_new_regions_list):
            blank_whole_w, blank_whole_h = judge_padding_value(merged_new_region.original_region, \
                blank_padding_type, blank_val)
            # print(f'blank_whole_w: {blank_whole_w}, blank_whole_h: {blank_whole_h}')
            # print(f'merged_region_w: {merged_new_region.w}, merged_region_h: {merged_new_region.h}')
            merged_region_id = merged_new_region.region_id
            # print(f'merged_region_id: {merged_region_id}')
            # print(f'merged_region_contain_list: {frame_merged_region_contain_dict[merged_region_id]}')
            merged_new_region = \
                pad_single_merged_region(merged_new_region, blank_whole_w, blank_whole_h, 
                frame_merged_region_contain_dict[merged_region_id], req_new_regions_dict, 
                region_image, context_image, tmp_regions_direc, 
                resize_type, area_upper_bound, fixed_ratio, fixed_area)
            # key-merged_region_id; value-merged_new_region
            if merged_new_region:
                merged_new_regions_dict[merged_region_id] = merged_new_region

        # update results
        merged_new_regions_contain_dict.update(frame_merged_region_contain_dict)
        cur_merged_region_id += len(merged_new_regions_list)

    return req_new_regions_dict, merged_new_regions_dict, merged_new_regions_contain_dict, \
        tmp_regions_direc, src_image_w, src_image_h

