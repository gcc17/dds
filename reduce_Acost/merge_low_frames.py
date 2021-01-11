import os
from math import (sqrt, ceil)
import cv2 as cv
from reduce_Bcost.streamB_utils import (normalize_image)
import numpy as np
from dds_utils import (Results, Region, calc_iou)
from .merge_region_class import (MergeLowRegion)
from rectpack import newPacker


def merge_frames(src_images_direc, save_images_direc, single_frame_cnt):
    os.makedirs(save_images_direc, exist_ok=True)
    width_cnt = int(sqrt(single_frame_cnt))
    height_cnt = int(ceil(single_frame_cnt / width_cnt))
    fnames = sorted([f for f in os.listdir(src_images_direc) if "png" in f])
    old_frame_cnt = len(fnames)
    new_frame_cnt = int(ceil(old_frame_cnt / single_frame_cnt))
    im = cv.imread(os.path.join(src_images_direc, fnames[0]))
    src_image_w = im.shape[1]
    src_image_h = im.shape[0]
    merged_image_w = src_image_w * width_cnt
    merged_image_h = src_image_h * height_cnt

    for i in range(new_frame_cnt):
        start_id = i*single_frame_cnt
        end_id = min(old_frame_cnt, (i+1)*single_frame_cnt)
        cur_fnames = fnames[start_id:end_id]
        new_frame = np.zeros((merged_image_h, merged_image_w, 3), dtype=np.uint8)
        new_frame = normalize_image(new_frame)

        for idx, cur_fname in enumerate(cur_fnames):
            src_frame = cv.imread(os.path.join(src_images_direc, cur_fname))
            src_frame = cv.cvtColor(src_frame, cv.COLOR_BGR2RGB)
            loc_x = int((idx % width_cnt) * src_image_w)
            loc_y = int((idx // width_cnt) * src_image_h)
            new_frame[loc_y:loc_y+src_image_h, loc_x:loc_x+src_image_w, :] = \
                src_frame
        new_frame = cv.cvtColor(new_frame, cv.COLOR_RGB2BGR)
        new_fname = f"{str(i).zfill(10)}.png"
        new_path = os.path.join(save_images_direc, new_fname)
        cv.imwrite(new_path, new_frame, [cv.IMWRITE_PNG_COMPRESSION, 0])


def restore_merged_results(single_frame_cnt, merged_results, start_fid):
    restored_results = Results()
    width_cnt = int(sqrt(single_frame_cnt))
    height_cnt = int(ceil(single_frame_cnt / width_cnt))
    width_single_frame = 1/width_cnt
    height_single_frame = 1/height_cnt

    for fid, regions_list in merged_results.regions_dict.items():
        for single_region in regions_list:
            for frame_id in range(single_frame_cnt):
                loc_x = (frame_id % width_cnt) * width_single_frame
                loc_y = (frame_id // width_cnt) * height_single_frame
                frame_region = Region(fid, loc_x, loc_y, width_single_frame, height_single_frame, 
                    conf=0.5, label='frame', resolution=0.8)
                if calc_iou(frame_region, single_region) > 0:
                    new_w = single_region.w * width_cnt
                    new_h = single_region.h * height_cnt
                    new_x = (single_region.x - loc_x) * width_cnt
                    new_y = (single_region.y - loc_y) * height_cnt
                    new_region = Region(fid*single_frame_cnt + frame_id + start_fid, 
                        new_x, new_y, new_w, new_h, 
                        single_region.conf, single_region.label, 
                        single_region.resolution, single_region.origin)
                    restored_results.append(new_region)
    
    return restored_results


def merge_frame_change_res(src_images_direc, regions_list, old_res, new_res, whole_res, 
        save_images_direc):
    os.makedirs(save_images_direc, exist_ok=True)
    src_fnames = sorted([f for f in os.listdir(src_images_direc) if "png" in f])
    im = cv.imread(os.path.join(src_images_direc, src_fnames[0]))
    src_image_w = im.shape[1]
    src_image_h = im.shape[0]
    merged_image_w = int(src_image_w * (whole_res // old_res))
    merged_image_h = int(src_image_h * (whole_res // old_res))
    packer = newPacker(rotation=False)
    packer.add_bin(width=merged_image_w, height=merged_image_h, count=float("inf"))

    for idx, single_region in enumerate(regions_list):
        abs_region_w = int(src_image_w * single_region.w * new_res / old_res)
        abs_region_h = int(src_image_h * single_region.h * new_res / old_res)
        packer.add_rect(width=abs_region_w, height=abs_region_h, rid=idx)
    
    print('before packing')
    packer.pack()
    all_rects = packer.rect_list()
    merged_images = {}
    merged_frames_maps = {}
    print('after packing')

    # after running rectpack, find out these rectangles
    prev_image = None
    prev_fid = -1
    for rect in all_rects:
        b, x, y, w, h, rid = rect
        if b not in merged_images.keys():
            merged_images[b] = np.zeros((merged_image_h, merged_image_w, 3), dtype=np.uint8)
            merged_images[b] = normalize_image(merged_images[b])
            merged_frames_maps[b] = []
        new_x = x / merged_image_w
        new_y = y / merged_image_h
        new_w = w / merged_image_w
        new_h = h / merged_image_h
        single_region = regions_list[rid]
        new_region_object = MergeLowRegion(
            single_region.x, single_region.y, single_region.w, single_region.h, 
            old_res, single_region.fid, 
            new_x, new_y, new_w, new_h, new_res)
        merged_frames_maps[b].append(new_region_object)
        if single_region.fid == prev_fid:
            cur_image = prev_image
        else:
            prev_fid = single_region.fid
            cur_fname = f"{str(single_region.fid).zfill(10)}.png"
            cur_image = cv.imread(os.path.join(src_images_direc, cur_fname))
            cur_image = cv.cvtColor(cur_image, cv.COLOR_BGR2RGB)
            prev_image = cur_image

        abs_old_x = int(src_image_w * single_region.x)
        abs_old_y = int(src_image_h * single_region.y)
        ori_abs_old_w = int(src_image_w * single_region.w)
        ori_abs_old_h = int(src_image_h * single_region.h)
        cur_region = cur_image[abs_old_y:abs_old_y+ori_abs_old_h, \
            abs_old_x:abs_old_x+ori_abs_old_w, :]
        resized_abs_old_w = int(src_image_w * single_region.w * new_res / old_res)
        resized_abs_old_h = int(src_image_h * single_region.h * new_res / old_res)
        cur_region = cv.resize(cur_region, (resized_abs_old_w, resized_abs_old_h), 
                            fx=0, fy=0, interpolation=cv.INTER_CUBIC)
        assert(w == resized_abs_old_w and h == resized_abs_old_h)
        merged_images[b][y:y+h, x:x+w, :] = cur_region

   # save merged_images
    for bid, merged_image in merged_images.items():
        merged_image_path = os.path.join(save_images_direc, f"{str(bid).zfill(10)}.png")
        merged_image = cv.cvtColor(merged_image, cv.COLOR_RGB2BGR)
        cv.imwrite(merged_image_path, merged_image, [cv.IMWRITE_PNG_COMPRESSION, 0])

    return merged_frames_maps


def restore_frame_change_res(merged_results, merged_frames_maps, 
        max_area_thresh=0.3, min_area_thresh=0.0):
    restored_results = Results()

    for bid, regions_list in merged_results.regions_dict.items():
        for single_region in regions_list:
            overlap_region_list = []
            if single_region.w * single_region.h > max_area_thresh:
                continue
            if single_region.w * single_region.h < min_area_thresh:
                continue
            for new_region in merged_frames_maps[bid]:
                cur_iou = calc_iou(single_region, new_region)
                if cur_iou > 0:
                    overlap_region_list.append((new_region, cur_iou))
            if len(overlap_region_list) == 0:
                continue
            # Sort by IoU, choose the largest
            overlap_region_list = sorted(overlap_region_list, key= lambda x:x[1])
            new_region = overlap_region_list[-1][0] 
            restored_x = max(single_region.x, new_region.x)
            restored_y = max(single_region.y, new_region.y)
            restored_w = min(single_region.x+single_region.w, new_region.x+new_region.w) - restored_x
            restored_h = min(single_region.y+single_region.h, new_region.y+new_region.h) - restored_y
            restored_w *= (new_region.old_res / new_region.new_res)
            restored_h *= (new_region.old_res / new_region.new_res)
            restored_x = new_region.old_x + \
                (restored_x-new_region.x) * (new_region.old_res / new_region.new_res)
            restored_y = new_region.old_y + \
                (restored_y-new_region.y) * (new_region.old_res / new_region.new_res)
            restored_results.append(
                Region(new_region.old_fid, restored_x, restored_y, restored_w, restored_h, 
                    single_region.conf, single_region.label, single_region.resolution))
    
    return restored_results

