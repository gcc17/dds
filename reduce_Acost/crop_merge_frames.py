import os
from math import (sqrt, ceil)
import cv2 as cv
from reduce_Bcost.streamB_utils import (normalize_image)
import numpy as np
from dds_utils import (Results, Region, calc_iou)


def crop_merge_frames_part(src_images_direc, save_images_direc, single_frame_cnt, 
        crop_x, crop_y, crop_w, crop_h):
    os.makedirs(save_images_direc, exist_ok=True)
    width_cnt = int(sqrt(single_frame_cnt))
    height_cnt = int(ceil(single_frame_cnt / width_cnt))
    fnames = sorted([f for f in os.listdir(src_images_direc) if "png" in f])
    old_frame_cnt = len(fnames)
    new_frame_cnt = int(ceil(old_frame_cnt / single_frame_cnt))
    im = cv.imread(os.path.join(src_images_direc, fnames[0]))
    crop_image_w = int(im.shape[1] * crop_w)
    crop_image_h = int(im.shape[0] * crop_h)
    crop_image_x = int(im.shape[1] * crop_x)
    crop_image_y = int(im.shape[0] * crop_y)
    merged_image_w = crop_image_w * width_cnt
    merged_image_h = crop_image_h * height_cnt

    for i in range(new_frame_cnt):
        start_id = i*single_frame_cnt
        end_id = min(old_frame_cnt, (i+1)*single_frame_cnt)
        cur_fnames = fnames[start_id:end_id]
        new_frame = np.zeros((merged_image_h, merged_image_w, 3), dtype=np.uint8)
        new_frame = normalize_image(new_frame)

        for idx, cur_fname in enumerate(cur_fnames):
            src_frame = cv.imread(os.path.join(src_images_direc, cur_fname))
            src_frame = cv.cvtColor(src_frame, cv.COLOR_BGR2RGB)
            src_frame = src_frame[crop_image_y:crop_image_y+crop_image_h, \
                crop_image_x:crop_image_x+crop_image_w, :]
            loc_x = int((idx % width_cnt) * crop_image_w)
            loc_y = int((idx // width_cnt) * crop_image_h)
            new_frame[loc_y:loc_y+crop_image_h, loc_x:loc_x+crop_image_w, :] = \
                src_frame
        new_frame = cv.cvtColor(new_frame, cv.COLOR_RGB2BGR)
        new_fname = f"{str(i).zfill(10)}.png"
        new_path = os.path.join(save_images_direc, new_fname)
        cv.imwrite(new_path, new_frame, [cv.IMWRITE_PNG_COMPRESSION, 0])