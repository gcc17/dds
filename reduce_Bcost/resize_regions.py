from .new_region import (PadShiftRegion)
import os
import ipdb
import cv2 as cv
import numpy as np
import shutil
from math import sqrt
from dds_utils import (Results)
import copy
from collections import OrderedDict
from .streamB_utils import (normalize_image, draw_region_rectangle)
    

def test_for_efficiency(target_region_path, src_image_w, src_image_h, \
    start_ratio, end_ratio, step_ratio, server, save_images_direc):
    from rectpack import newPacker
    target_region = cv.imread(target_region_path)
    target_region = cv.cvtColor(target_region, cv.COLOR_BGR2RGB)
    os.makedirs(save_images_direc, exist_ok=True)
    
    resize_ratio_list = []
    resize_region_list = []
    for resize_ratio in np.arange(start_ratio, end_ratio, step_ratio):
        resize_ratio_list.append(resize_ratio)
        new_region = cv.resize(target_region, (0, 0), fx=resize_ratio, fy=resize_ratio, \
            interpolation=cv.INTER_NEAREST)
        resize_region_list.append(new_region)

    packer = newPacker(rotation=False)
    packer.add_bin(width=src_image_w, height=src_image_h, count=float("inf"))
    for idx, new_region in enumerate(resize_region_list):
        resize_ratio = resize_ratio_list[idx]
        region_w = new_region.shape[1]
        region_h = new_region.shape[0]
        packer.add_rect( width=region_w, height=region_h, rid=idx )
    packer.pack()
    all_rects = packer.rect_list()
    shift_images = {}
    fnames = []

    # after running rectpack, find out these rectangles
    print(save_images_direc)
    for rect in all_rects:
        b, x, y, w, h, rid = rect
        if b not in shift_images.keys():
            shift_images[b] = np.zeros((src_image_h, src_image_w, 3), dtype=np.uint8)
            shift_images[b] = (shift_images[b])
            fnames.append(f"{str(b).zfill(10)}.png")
        shift_images[b][y:y+h, x:x+w, :] = resize_region_list[rid]
            
    for b, shift_image in shift_images.items():
        shift_image_path = os.path.join(save_images_direc, f"{str(b).zfill(10)}.png")
        shift_image = cv.cvtColor(shift_image, cv.COLOR_RGB2BGR)
        shift_images[b] = shift_image
        cv.imwrite(shift_image_path, shift_image, [cv.IMWRITE_PNG_COMPRESSION, 0])
    
    final_results, rpn_regions = \
        server.perform_detection(None, 0.8, fnames=fnames, images=shift_images)
    draw_region_rectangle(save_images_direc, fnames, final_results.regions_dict, 
        f'{save_images_direc}-vis', display_result=True)
    shutil.rmtree(save_images_direc)


def update_resize_ratios(cur_resize_ratio_list, increase_ratio):
    if len(cur_resize_ratio_list) == 1:
        if increase_ratio:
            return [2, 3]
        else:
            return cur_resize_ratio_list
    if len(cur_resize_ratio_list) == 2:
        if increase_ratio:
            return [2, 2.5, 3]
        else:
            return [2]
    if len(cur_resize_ratio_list) == 3:
        if increase_ratio:
            return [2, 2.5, 3, 3.5]
        else:
            return [2, 3]
    else:
        if increase_ratio:
            return cur_resize_ratio_list
        else:
            return [2, 2.5, 3]
    
