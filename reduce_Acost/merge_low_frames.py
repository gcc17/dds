import os
from math import (sqrt, ceil)
import cv2 as cv
from reduce_Bcost.streamB_utils import (normalize_image)
import numpy as np
from dds_utils import (Results, Region, calc_iou)


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


