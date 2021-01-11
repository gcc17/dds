from dds_utils import (read_results_dict, Region)
from reduce_Bcost.streamB_utils import (normalize_image)
from reduce_Bcost.new_region import (PadShiftRegion)
import cv2 as cv
import numpy as np
import os

def read_object_regions(regions_dict, region_enlarge_ratio, fnames, 
        src_images_direc, save_direc, max_obj_size=0.3):
    os.makedirs(save_direc, exist_ok=True)
    region_map_dict = {}
    for fname in fnames:
        fid = int(fname.split(".")[0])
        if fid not in regions_dict.keys():
            continue
        regions_list = regions_dict[fid]
        src_frame = cv.imread(os.path.join(src_images_direc, fname))
        src_frame = cv.cvtColor(src_frame, cv.COLOR_BGR2RGB)
        width = src_frame.shape[1]
        height = src_frame.shape[0]
        save_frame = np.zeros_like(src_frame)
        save_frame = normalize_image(save_frame)
        frame_map = np.zeros((height, width), dtype=np.uint8)

        for single_region in regions_list:
            if single_region.w * single_region.h > max_obj_size:
                continue
            single_region.enlarge(region_enlarge_ratio)
            x0 = int(single_region.x * width)
            y0 = int(single_region.y * height)
            x1 = x0 + int(single_region.w * width)
            y1 = y0 + int(single_region.h * height)
            save_frame[y0:y1, x0:x1, :] = src_frame[y0:y1, x0:x1, :]
            frame_map[y0:y1, x0:x1] = 1

        save_frame = cv.cvtColor(save_frame, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(save_direc, fname), save_frame,
                   [cv.IMWRITE_PNG_COMPRESSION, 0])
        region_map_dict[str(fid)] = frame_map
    map_fname = "region_map.npz"
    np.savez(os.path.join(save_direc, map_fname), **region_map_dict)

    return region_map_dict


def get_largest_bbox(region_map_dict, src_images_direc, box_id_start=1, 
        npz_save_direc=None, save_images_direc=None, box_color=(255,0,0)):
    if save_images_direc:
        os.makedirs(save_images_direc, exist_ok=True)
    all_bboxes = {}
    region_id_map_dict = {}
    region_area_dict = {}
    src_image_w = 0
    src_image_h = 0
    cur_box_id = box_id_start

    for fid, frame_map in region_map_dict.items():
        fid = str(fid)
        fname = f"{str(fid).zfill(10)}.png"
        src_frame = cv.imread(os.path.join(src_images_direc, fname))
        src_frame = cv.cvtColor(src_frame, cv.COLOR_BGR2RGB)
        width = src_frame.shape[1]
        height = src_frame.shape[0]
        if not src_image_w:
            src_image_w = width
            src_image_h = height
        assert(width == src_image_w and height == src_image_h)
        region_id_map_dict[fid] = np.zeros((height, width))
        all_bboxes[fid] = []

        # Find contours for overlaped regions
        contours, hierarchy = \
		    cv.findContours(frame_map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
        for single_contour in contours:
            bbox = cv.boundingRect(single_contour)
            x,y,w,h = bbox
            if save_images_direc:
                cv.rectangle(src_frame, (x, y), (x+w, y+h), box_color, 2)
            
            # Mark pixels inside these regions
            region_area_dict[cur_box_id] = 0
            for cur_x in range(x, x+w):
                for cur_y in range(y, y+h):
                    if cv.pointPolygonTest(single_contour, (cur_x, cur_y), False) >= 0:
                        region_id_map_dict[fid][cur_y, cur_x] = cur_box_id
                        region_area_dict[cur_box_id] += 1
            # Change bbox coordinates to relative value
            x0 = x / width
            y0 = y / height
            w = w / width
            h = h / height
            all_bboxes[fid].append((x0, y0, w, h, cur_box_id))
            cur_box_id += 1
        
        if save_images_direc:
            src_frame = cv.cvtColor(src_frame, cv.COLOR_RGB2BGR)
            cv.imwrite(os.path.join(save_images_direc, fname), src_frame,
                    [cv.IMWRITE_PNG_COMPRESSION, 0])
    if npz_save_direc:
        map_fname = "region_id_map.npz"
        np.savez(os.path.join(npz_save_direc, map_fname), **region_id_map_dict)

    return all_bboxes, region_id_map_dict, region_area_dict, src_image_w, src_image_h


def wrap_bboxes(all_bboxes, resolution=0.8):
    all_new_regions_dict = {}
    for fid, bboxes in all_bboxes.items():
        for bbox in bboxes:
            x,y,w,h,cur_box_id = bbox
            ori_region = Region(fid, x, y, w, h, 1.0, label="object", resolution=resolution)
            new_region = PadShiftRegion(ori_region, cur_box_id, x, y, w, h)
            all_new_regions_dict[cur_box_id] = new_region
    
    return all_new_regions_dict

