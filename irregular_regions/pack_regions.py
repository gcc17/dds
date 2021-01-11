import os
from reduce_Bcost.streamB_utils import (normalize_image, find_region_same_id, draw_region_rectangle)
from dds_utils import (Results, Region)
import cv2 as cv
import numpy as np
from rectpack import newPacker
from rectpack.guillotine import GuillotineBssfSas
import shutil


def pack_bboxes(
        all_new_regions_dict, region_id_map_dict, region_images_direc, pack_images_direc, 
        src_image_w, src_image_h, start_bid=0, 
    ):
    
     # Create directory for saving region-packed frames
    os.makedirs(pack_images_direc, exist_ok=True)
    for fname in os.listdir(pack_images_direc):
        if 'png' in fname:
            os.remove(os.path.join(pack_images_direc, fname))

    # Create packer of rectpack
    # packer = newPacker(pack_algo=GuillotineBssfSas, rotation=False)
    packer = newPacker(rotation=False)
    packer.add_bin(width=src_image_w, height=src_image_h, count=float("inf"))

    # add new region(bbox) into the packer
    for cur_new_region_id, cur_new_region in all_new_regions_dict.items():
        abs_total_w = int(cur_new_region.w * src_image_w)
        abs_total_h = int(cur_new_region.h * src_image_h)
        packer.add_rect(
            width=abs_total_w, height=abs_total_h, rid=cur_new_region_id
        )
        # print(f'({abs_total_w}, {abs_total_h})')
    
    print('before packing')
    packer.pack()
    all_rects = packer.rect_list()
    pack_images = {}
    pack_regions_map = {}
    print('after packing')
    
    last_fid = -1
    last_frame = None
    # after running rectpack, find out these rectangles
    for rect in all_rects:
        b, x, y, w, h, rid = rect
        b += start_bid
        if b not in pack_images.keys():
            pack_images[b] = np.zeros((src_image_h, src_image_w, 3), dtype=np.uint8)
            pack_regions_map[b] = np.zeros((src_image_h, src_image_w), dtype=int)
            # set the whole image as single color
            pack_images[b] = normalize_image(pack_images[b])
        
        if rid not in all_new_regions_dict.keys():
            exit()
        cur_new_region = all_new_regions_dict[rid]

        # Get original region location
        abs_ori_x = int(cur_new_region.original_region.x * src_image_w)
        abs_ori_y = int(cur_new_region.original_region.y * src_image_h)
        fid = cur_new_region.original_region.fid
        pack_regions_map[b][y:y+h, x:x+w] = \
            region_id_map_dict[str(fid)][abs_ori_y:abs_ori_y+h, abs_ori_x:abs_ori_x+w]
        
        # Update region location
        all_new_regions_dict[rid].x = x/src_image_w
        all_new_regions_dict[rid].y = y/src_image_h

        # Read region content from file
        if fid == last_fid:
            cur_frame = last_frame
        else:
            cur_frame_path = os.path.join(region_images_direc, f"{str(fid).zfill(10)}.png")
            cur_frame = cv.imread(cur_frame_path)
            cur_frame = cv.cvtColor(cur_frame, cv.COLOR_BGR2RGB)
            last_fid = fid
            last_frame = cur_frame
        
        # Set region_content in pack_images
        pack_images[b][y:y+h, x:x+w, :] = cur_frame[abs_ori_y:abs_ori_y+h, abs_ori_x:abs_ori_x+w, :]
    
    # save pack_images
    for bid, pack_image in pack_images.items():
        pack_image_path = os.path.join(pack_images_direc, f"{str(bid).zfill(10)}.png")
        pack_image = cv.cvtColor(pack_image, cv.COLOR_RGB2BGR)
        cv.imwrite(pack_image_path, pack_image, [cv.IMWRITE_PNG_COMPRESSION, 0])
    
    # cleanup temporary region images
    # shutil.rmtree(region_images_direc) 
    
    return pack_regions_map



