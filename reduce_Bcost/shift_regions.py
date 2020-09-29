import os
from .streamB_utils import (normalize_image, find_region_same_id)
import ipdb
import cv2 as cv
import numpy as np
from rectpack import newPacker
from rectpack.guillotine import GuillotineBssfSas
import shutil


def pack_merged_regions_as_image(merged_new_regions_dict, \
        req_new_regions_dict, merged_new_regions_contain_dict, \
        shift_image_direc, tmp_regions_direc, src_image_w, src_image_h):
    
    # create directory for saving shift_images
    os.makedirs(shift_image_direc, exist_ok=True)
    for fname in os.listdir(shift_image_direc):
        if 'png' in fname:
            os.remove(os.path.join(shift_image_direc, fname))
    # create packer of rectpack
    packer = newPacker(pack_algo=GuillotineBssfSas, rotation=False)
    packer.add_bin(width=src_image_w, height=src_image_h, count=float("inf"))

    # add context_blank-padded merged region into to the packer
    for cur_merged_new_region in merged_new_regions_dict.values():
        abs_total_w = int(cur_merged_new_region.w * src_image_w)
        abs_total_h = int(cur_merged_new_region.h * src_image_h)
        packer.add_rect(
            width=abs_total_w, height=abs_total_h, rid=cur_merged_new_region.region_id
        )
    
    print('before packing')
    packer.pack()
    all_rects = packer.rect_list()
    shift_images = {}
    merged_regions_maps = {}
    print('after packing')
    
    # after running rectpack, find out these rectangles
    for rect in all_rects:
        b, x, y, w, h, rid = rect
        if b not in shift_images.keys():
            shift_images[b] = np.zeros((src_image_h, src_image_w, 3), dtype=np.uint8)
            merged_regions_maps[b] = np.zeros((src_image_h, src_image_w), dtype=int)
            merged_regions_maps[b][:,:] = -1
            # set the whole image as 'blank'
            shift_images[b] = normalize_image(shift_images[b])
        
        if rid not in merged_new_regions_dict.keys():
            exit()
        cur_merged_new_region = merged_new_regions_dict[rid]
        # get shift amount: new_location - old_location
        shift_x = x/src_image_w - cur_merged_new_region.x
        shift_y = y/src_image_h - cur_merged_new_region.y
        
        # read region content from file
        full_pad_region_path = os.path.join(tmp_regions_direc, f"region-{rid}.png")
        region_content = cv.imread(full_pad_region_path)
        region_content = cv.cvtColor(region_content, cv.COLOR_BGR2RGB)
        # set region_content in shift_images
        shift_images[b][y:y+h, x:x+w, :] = region_content

        # set merged_regions_maps, used when restoring detection results back to original image
        # this map is only used to find regions that overlap with the detection box
        # box calculate IoU with small new region
        merged_regions_maps[b][y:y+h, x:x+w] = rid
        
        # shift small new regions inside this merged new region
        for cur_req_new_region_id in merged_new_regions_contain_dict[rid]:
            cur_req_new_region = req_new_regions_dict[cur_req_new_region_id]
            cur_req_new_region.x += shift_x
            cur_req_new_region.y += shift_y
            req_new_regions_dict[cur_req_new_region_id] = cur_req_new_region
    
    # save shift_images
    for bid, shift_image in shift_images.items():
        shift_image_path = os.path.join(shift_image_direc, f"{str(bid).zfill(10)}.png")
        shift_image = cv.cvtColor(shift_image, cv.COLOR_RGB2BGR)
        cv.imwrite(shift_image_path, shift_image, [cv.IMWRITE_PNG_COMPRESSION, 0])
    
    # cleanup temporary region images
    # shutil.rmtree(tmp_regions_direc) 
    
    return merged_regions_maps
