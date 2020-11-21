import cv2 as cv
import os
from dds_utils import (Results, Region)
from .streamB_utils import (region_iou, filter_true_regions)

def filter_regions_dds(dds_results, cur_req_regions_result, dds_fid, low_images_path, 
        match_thresh=0.9, filter_iou=0.0, template_method=cv.TM_CCOEFF_NORMED, 
        confid_thresh=0.5, max_object_size=0.3):
    dds_image_path = os.path.join(low_images_path, f"{str(dds_fid).zfill(10)}.png")
    dds_image = cv.imread(dds_image_path, 0)
    filtered_req_regions_result = Results()
    infer_regions_list = filter_true_regions(
        dds_results.regions_dict[dds_fid], confid_thresh, max_object_size)

    for fid in cur_req_regions_result.regions_dict.keys():
        regions_image_path = os.path.join(low_images_path, f"{str(fid).zfill(10)}.png")
        regions_image = cv.imread(regions_image_path, 0)
        image_width = regions_image.shape[1]
        image_height = regions_image.shape[0]
        for single_region in cur_req_regions_result.regions_dict[fid]:
            region_x0 = int(single_region.x * image_width)
            region_y0 = int(single_region.y * image_height)
            region_x1 = int(single_region.w * image_width) + region_x0
            region_y1 = int(single_region.h * image_height) + region_y0
            single_region_image = regions_image[region_y0:region_y1, region_x0:region_x1]

            # matchTemplate(image, template, method)
            res = cv.matchTemplate(dds_image, single_region_image, template_method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            if template_method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                match_val = 1 - min_val
                # match_loc[0]: w, match_loc[1]: h
                match_loc = min_loc
            else:
                match_val = max_val
                match_loc = max_loc
            
            if match_val < match_thresh:
                filtered_req_regions_result.append(single_region)
                continue
            # find match location in the dds frame
            match_region = Region(fid, match_loc[0]/image_width, match_loc[1]/image_height,
                single_region.w, single_region.h, 
                single_region.conf, single_region.label, single_region.resolution)
            # check if match location has detection box
            for infer_region in infer_regions_list:
                match_iou = region_iou(infer_region, match_region)
                if match_iou > filter_iou:
                    filtered_req_regions_result.append(single_region)
                    break
    
    return filtered_req_regions_result

