import os
import logging
import argparse
from dds_utils import (merge_boxes_in_results, Results, compute_regions_size, Region, \
    read_results_dict)
from .merge_regions import (encode_regions_images, encode_batch_filtered_regions)
from .pad_regions import (pad_filtered_regions_context, pad_filtered_regions_blank)
from .shift_regions import (pack_filtered_padded_regions)
from .restore_results import (restore_merged_regions_detection_box)
from .streamB_utils import (draw_region_rectangle, write_compute_cost)
from .filter_regions import (filter_regions_dds)
import ipdb
import cv2 as cv
import numpy as np
import shutil
import timeit


def analyze_dds_filtered(
        server, video_name, req_regions_result, low_results_path,
        context_padding_type, context_val, blank_padding_type, blank_val,
        intersect_iou, merge_iou, filter_by_dds,
        cleanup, out_cost_file=None
    ):

    logger = logging.getLogger("streamB-dds-filtered")
    handler = logging.NullHandler()
    logger.addHandler(handler)
    logger.info(f'{video_name} initialize')

    low_results_dict = None
    if low_results_path:
        low_results_dict = read_results_dict(low_results_path)

    total_size = [0, 0]
    total_time = 0
    total_pad_time = 0
    total_shift_time = 0
    total_infer_time = 0
    total_frame_cnt = 0
    all_restored_results = Results()
    dnn_output_results = Results()
    high_phase_results = Results()
    fnames = sorted([f for f in os.listdir(server.config.high_images_path) if "png" in f])
    number_of_frames = len(list(fnames))
    start_bid = 0

    for i in range(0, number_of_frames, server.config.batch_size):
        start_fid = i
        end_fid = min(number_of_frames, i + server.config.batch_size)
        # req_regions in the batch except the first frame
        cur_req_regions_result = Results()
        for fid in range(start_fid+1, end_fid):
            if fid in req_regions_result.regions_dict.keys():
                for single_region in req_regions_result.regions_dict[fid]:
                    cur_req_regions_result.append(single_region)
        # Run dds on the first frame in this batch
        dds_req_regions_result = Results()
        if start_fid in req_regions_result.regions_dict.keys():
            for single_region in req_regions_result.regions_dict[start_fid]:
                dds_req_regions_result.append(single_region)
        if not filter_by_dds:
            cur_req_regions_result.combine_results(dds_req_regions_result, 1)

        logger.info(f"Processing batch from {start_fid} to {end_fid}")
        batch_images_direc = f'{video_name}-{start_fid}-{end_fid}'

        # Encode low-quality images
        base_req_regions = Results()
        for fid in range(start_fid, end_fid):
            base_req_regions.append(
                Region(fid, 0, 0, 1, 1, 1.0, 2, server.config.high_resolution))
        base_images_direc = f'{batch_images_direc}-base-phase'
        low_batch_video_size, low_batch_pixel_size = encode_regions_images(
            base_req_regions, server.config.high_images_path, \
            server.config.low_resolution, server.config.low_qp, \
            base_images_direc, server.config.enforce_iframes, True
        )
        total_size[0] += low_batch_video_size
        r1, req_regions = server.simulate_low_query(
            start_fid, end_fid, f"{base_images_direc}-cropped", low_results_dict, False,
            server.config.rpn_enlarge_ratio)
        all_restored_results.combine_results(
            r1, server.config.intersection_threshold)

        # Run dds on the first frame
        if filter_by_dds:
            dds_images_direc = f'{batch_images_direc}-dds'
            dds_regions_size, _ = compute_regions_size(
                dds_req_regions_result, dds_images_direc, server.config.high_images_path,
                server.config.high_resolution, server.config.high_qp,
                server.config.enforce_iframes, True)
            logger.info(f"Sent {len(dds_req_regions_result)} regions which have "
                        f"{dds_regions_size / 1024}KB in dds phase using {server.config.high_qp}")
            total_size[1] += dds_regions_size

            dds_results, infer_elapsed = server.emulate_high_query(
                dds_images_direc, f'{base_images_direc}-cropped', dds_req_regions_result)
            total_infer_time += infer_elapsed
            total_frame_cnt += 1

            # Filter req_region in the batch
            cur_req_regions_result = filter_regions_dds(
                dds_results, cur_req_regions_result, start_fid,f'{base_images_direc}-cropped',
                match_thresh=0.9, filter_iou=0.0, 
                confid_thresh=server.config.prune_score, max_object_size=server.config.max_object_size
            )

        if len(cur_req_regions_result) == 0:
            continue
        # Merge high-quality req_regions and low-quality frames
        merged_images_direc = f'{batch_images_direc}-merged'
        high_batch_video_size, _ = encode_batch_filtered_regions(
            logger, cur_req_regions_result, server.config.high_qp, server.config.high_resolution,
            server.config.high_images_path, f'{base_images_direc}-cropped', merged_images_direc, 
            server.config.enforce_iframes
        )
        total_size[1] += high_batch_video_size

        # Pad and resize regions
        pad_start = timeit.default_timer()
        # First do context padding and merge regions
        req_new_regions_dict, merged_new_regions_dict, merged_new_regions_contain_dict = \
            pad_filtered_regions_context(
                cur_req_regions_result, context_padding_type, context_val, merge_iou
            )
        # Then do blank paddding
        padded_regions_direc, src_image_w, src_image_h, merged_new_regions_dict = \
            pad_filtered_regions_blank(merged_new_regions_dict, batch_images_direc, 
            f'{merged_images_direc}-cropped', blank_padding_type, blank_val, 
            sort_context_region=True)
        pad_elapsed = (timeit.default_timer() - pad_start)
        total_pad_time += pad_elapsed
        logger.info(f"{video_name} finish padding")

        # Shift padded regions
        shift_start = timeit.default_timer()
        shift_images_direc = f'{batch_images_direc}-pack'
        merged_regions_maps = pack_filtered_padded_regions(
            req_new_regions_dict, merged_new_regions_dict, merged_new_regions_contain_dict, 
            shift_images_direc, padded_regions_direc, src_image_w, src_image_h, start_bid=start_bid)

        shift_elapsed = (timeit.default_timer() - shift_start)
        total_shift_time += shift_elapsed
        frame_cnt = len(sorted(os.listdir(shift_images_direc)))
        start_bid += frame_cnt
        total_frame_cnt += frame_cnt
        logger.info(f"{video_name} finish shifting")

        # Perform detection
        infer_start = timeit.default_timer()
        pad_shift_results, rpn_results = server.perform_detection(shift_images_direc, \
            server.config.high_resolution)
        infer_elapsed = (timeit.default_timer() - infer_start)
        total_infer_time += infer_elapsed
        logger.info(f'{video_name} finish detection')
        dnn_output_results.combine_results(pad_shift_results, server.config.intersection_threshold)

        # Restore results
        restore_start = timeit.default_timer()
        restored_pad_shift_results, restored_req_regions_id = \
            restore_merged_regions_detection_box(
                pad_shift_results, merged_new_regions_dict, 
                req_new_regions_dict, merged_new_regions_contain_dict, 
                merged_regions_maps, src_image_w, src_image_h, iou_thresh=intersect_iou
            )
        
        if not cleanup:
            vis_pack_direc = f'{video_name}-vis-pack'
            pack_fnames = sorted([f for f in os.listdir(shift_images_direc) if "png" in f])
            draw_region_rectangle(shift_images_direc, pack_fnames, pad_shift_results.regions_dict, 
                vis_pack_direc, display_result=True, drop_no_rect=False, clean_save=False)
            vis_restore_direc = video_name + '-vis-restore'
            restore_fnames = []
            for cur_fid in range(start_fid, end_fid):
                restore_fnames.append(f"{str(cur_fid).zfill(10)}.png")
            draw_region_rectangle(server.config.high_images_path, restore_fnames, \
                restored_pad_shift_results.regions_dict, vis_restore_direc, \
                display_result=True, clean_save=False, drop_no_rect=True)

        # Combine with results on packed images
        all_restored_results.combine_results(restored_pad_shift_results, 
            server.config.intersection_threshold)
        high_phase_results.combine_results(restored_pad_shift_results, 
            server.config.intersection_threshold)
        if filter_by_dds:
            # Combine with results on DDS image
            all_restored_results.combine_results(dds_results, 
                server.config.intersection_threshold)
            high_phase_results.combine_results(dds_results, 
                server.config.intersection_threshold)
        restore_elapsed = (timeit.default_timer() - restore_start)
        total_time += restore_elapsed

        # Remove related directory
        shutil.rmtree(shift_images_direc)
        shutil.rmtree(f'{merged_images_direc}-cropped')
        shutil.rmtree(f'{base_images_direc}-cropped')
        if filter_by_dds:
            shutil.rmtree(f'{dds_images_direc}-cropped')

    dnn_output_results.write(f"{video_name}-dnn")
    high_phase_results.write(f"{video_name}-high")
    all_restored_results.write(f"{video_name}-ori")

    # merge detection box again
    all_restored_results = merge_boxes_in_results(all_restored_results.regions_dict, 
        server.config.low_threshold, server.config.suppression_threshold)
    all_restored_results.write(f"{video_name}")
    total_time += total_pad_time
    total_time += total_shift_time
    total_time += total_infer_time
    logger.info(f"Padding time {total_pad_time}, shifting time {total_shift_time}, "
                f"infer time {total_infer_time} for {total_frame_cnt} frames, "
                f"total elapsed time {total_time}")
    if out_cost_file:
        write_compute_cost(out_cost_file, video_name, total_frame_cnt, 
            total_pad_time, total_shift_time, total_infer_time, total_time)
    
    # Fill empty frames
    all_restored_results.fill_gaps(number_of_frames)

    return all_restored_results, total_size


