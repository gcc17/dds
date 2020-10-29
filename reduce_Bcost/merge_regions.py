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


def get_req_regions_group(req_regions_result, area_list, percent_list, idx):
    if idx == 0:
        lower_bound = 0.0
    else:
        lower_bound = get_area_percentile(area_list, percent_list[idx-1])
    upper_bound = get_area_percentile(area_list, percent_list[idx])

    cur_req_regions_result = Results()
    for fid, region_list in req_regions_result.regions_dict.items():
        has_region = False
        for single_region in region_list:
            region_area = single_region.w * single_region.h
            if lower_bound < region_area <= upper_bound:
                cur_req_regions_result.append(single_region)
                has_region = True
        if not has_region:
            cur_req_regions_result.append(Region(fid, 0, 0, 0, 0, 0.1, "no obj", 1.0))
    
    return cur_req_regions_result


def encode_regions_images(req_regions_result, high_images_path, resolution, qp, save_images_direc, \
        enforce_iframes=True, estimate_bandwidth=True):

    encoded_batch_video_size, batch_pixel_size = compute_regions_size(
        req_regions_result, save_images_direc, high_images_path,
        resolution, qp, enforce_iframes, estimate_bandwidth)
    extract_images_direc = f'{save_images_direc}-cropped'
    extract_images_from_video(extract_images_direc, req_regions_result)

    return encoded_batch_video_size, batch_pixel_size


def encode_regions_strategy(req_regions_result, high_images_path, resolution, \
        percent_list, qp_list, encode_images_direc, enforce_iframes=True, estimate_bandwidth=True):
    area_list = get_area_list(req_regions_result)
    
    encoded_video_size = 0.0
    pixel_size = 0.0
    for idx in range(len(percent_list)):
        # Select req_regions whose area between (percent_list[idx-1], percent_list[idx]]
        cur_req_regions_result = get_req_regions_group(req_regions_result, area_list, percent_list, idx)
        # Encode req_regions_result 
        region_images_direc = f'{encode_images_direc}-{idx}'
        encoded_batch_video_size, batch_pixel_size = encode_regions_images(
            cur_req_regions_result, high_images_path, resolution, qp_list[idx], \
            region_images_direc, enforce_iframes, estimate_bandwidth
        )
        encoded_video_size += encoded_batch_video_size
        pixel_size += batch_pixel_size
    
    return encoded_video_size, pixel_size


def merge_low_encode_images(base_images_direc, encode_images_direc, percent_list, merged_images_direc, \
        req_regions_result):
    base_images_direc = f'{base_images_direc}-cropped'
    area_list = get_area_list(req_regions_result)
    # Larger percent use lower-quality qp
    # Reverse the encoded region: add high-quality region above low-quality region
    for idx in range(len(percent_list)-1, -1, -1):
        cur_req_regions_result = get_req_regions_group(req_regions_result, area_list, percent_list, idx)
        # print(len(cur_req_regions_result.regions_dict), cur_req_regions_result.regions_dict.keys())
        cur_region_images_direc = f'{encode_images_direc}-{idx}-cropped'
        last_region_images_direc = f'{encode_images_direc}-{idx+1}-cropped'
        import ipdb; ipdb.set_trace()
        if idx == len(percent_list)-1:
            merge_images(cur_region_images_direc, base_images_direc, cur_req_regions_result)
        else:
            merge_images(cur_region_images_direc, last_region_images_direc, cur_req_regions_result)

    # Copy merged images to the dst directory
    os.makedirs(merged_images_direc, exist_ok=True)
    fnames = sorted([f for f in os.listdir(cur_region_images_direc) if "png" in f])
    for img in fnames:
        shutil.copy(os.path.join(cur_region_images_direc, img), merged_images_direc)
    
    # Cleanup tmp regions directory
    shutil.rmtree(base_images_direc)
    for idx in range(len(percent_list)):
        region_images_direc = f'{encode_images_direc}-{idx}-cropped'
        shutil.rmtree(region_images_direc)


def encode_regions(server, logger, req_regions_result, percent_list, qp_list, batch_size, \
        high_images_path, merged_images_direc):
    total_start = min(list(req_regions_result.regions_dict.keys()))
    total_end = max(list(req_regions_result.regions_dict.keys())) + 1
    total_size = [0, 0]

    for i in range(total_start, total_end, batch_size):
        start_fid = i
        end_fid = min(total_end, i + batch_size)
        logger.info(f"Processing batch from {start_fid} to {end_fid}")

        # Encode frames in batch and get size
        # Make temporary frames to downsize complete frames
        base_req_regions = Results()
        for fid in range(start_fid, end_fid):
            base_req_regions.append(
                Region(fid, 0, 0, 1, 1, 1.0, 2, server.config.high_resolution))
        base_images_direc = f'{merged_images_direc}-base-phase'
        encoded_batch_video_size, batch_pixel_size = encode_regions_images(
            base_req_regions, high_images_path, server.config.low_resolution, server.config.low_qp, \
            base_images_direc, server.config.enforce_iframes, True
        )
        logger.info(f"Sent {encoded_batch_video_size / 1024} in base phase")
        total_size[0] += encoded_batch_video_size

        # Encode regions in batch and get size
        # Regions with different size are encoded with different qp
        cur_req_regions_result = Results()
        for fid in range(start_fid, end_fid):
            if fid in req_regions_result.regions_dict.keys():
                for single_region in req_regions_result.regions_dict[fid]:
                    cur_req_regions_result.append(single_region)
        encode_images_direc = f'{merged_images_direc}-encode'
        encoded_video_size, pixel_size = encode_regions_strategy(
            cur_req_regions_result, high_images_path, server.config.high_resolution, \
            percent_list, qp_list, encode_images_direc, server.config.enforce_iframes, True
        )
        total_size[1] += encoded_video_size

        # Merge base images and encoded regions
        merge_low_encode_images(
            base_images_direc, encode_images_direc, percent_list, merged_images_direc, \
            cur_req_regions_result
        )
    
    return total_size


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
