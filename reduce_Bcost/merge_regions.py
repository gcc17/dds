import numpy as np
import os
from dds_utils import (Results, compute_regions_size, Region, extract_images_from_video, \
    merge_images)
import shutil


def get_area_percentile(area_list, req_percent):
    area_np = np.array(area_list)
    if 0 <= req_percent <= 1:
        req_percent *= 100
    if req_percent < 0 or req_percent > 100:
        print('Error percent')
        return None
    return np.percentile(area_np, req_percent)


def get_area_list(req_regions_result):
    area_list = []
    for fid, region_list in req_regions_result.regions_dict.items():
        for single_region in region_list:
            area_list.append(single_region.w * single_region.h)
    return area_list


def encode_regions_images(req_regions_result, high_images_path, resolution, qp, save_images_direc, \
        enforce_iframes=True, estimate_bandwidth=True):

    encoded_batch_video_size, batch_pixel_size = compute_regions_size(
        req_regions_result, save_images_direc, high_images_path,
        resolution, qp, enforce_iframes, estimate_bandwidth)
    extract_images_direc = f'{save_images_direc}-cropped'
    extract_images_from_video(extract_images_direc, req_regions_result)

    return encoded_batch_video_size, batch_pixel_size


def encode_batch_filtered_regions(
        logger, batch_req_regions_result, qp, high_resolution, 
        high_images_path, low_images_direc, merged_images_direc, enforce_iframes
    ):

    high_batch_video_size, high_batch_pixel_size = encode_regions_images(
        batch_req_regions_result, high_images_path, high_resolution, \
        qp, merged_images_direc, enforce_iframes, True
    )
    logger.info(f"Sent {high_batch_video_size / 1024} in high phase")
    merge_images(f'{merged_images_direc}-cropped', low_images_direc, batch_req_regions_result)

    return high_batch_video_size, high_batch_pixel_size
