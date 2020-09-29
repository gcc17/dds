import os
import logging
import argparse
from dds_utils import (visualize_regions, merge_boxes_in_results)
from .pad_regions import (pad_all_req_regions)
from .shift_regions import (pack_merged_regions_as_image)
from .restore_results import (restore_detection_box, compress_restored_results, 
        restore_merged_regions_detection_box, restore_track_regions_detection_box)
from .streamB_utils import (get_percentile, two_results_diff, draw_region_rectangle, 
        filter_true_regions, write_compute_cost)
from .region_similarity import (track_similar_region)
import ipdb
import cv2 as cv
import numpy as np
import shutil
import timeit


def analyze_merge_regions(
        server, video_name, region_image_direc, context_image_direc, high_resolution,
        req_regions_result, mpeg_regions_result, 
        context_padding_type, context_val, blank_padding_type, blank_val,
        cleanup=False, gt_regions_dict=None,
        out_cost_file=None, 
        intersect_iou=0.2,
        resize_type='fixed_ratio', area_upper_bound=0.0005, fixed_ratio=1, fixed_area=1, 
        merge_iou=0.0, track_region=False
    ):

    logger = logging.getLogger("streamB-resize-pad-shift")
    handler = logging.NullHandler()
    logger.addHandler(handler)
    logger.info(f"{video_name} initialize")

    total_time = 0

    # pad regions
    pad_start = timeit.default_timer()
    req_new_regions_dict, merged_new_regions_dict, merged_new_regions_contain_dict, \
    tmp_regions_direc, src_image_w, src_image_h = \
        pad_all_req_regions(req_regions_result, 
            region_image_direc, context_image_direc, video_name,
            context_padding_type, context_val, blank_padding_type, blank_val, 
            resize_type, area_upper_bound, fixed_ratio, fixed_area, merge_iou
        )
    if track_region:
        track_new_regions_dict, track_new_regions_contain_dict = \
            track_similar_region(merged_new_regions_dict, tmp_regions_direc)

    pad_elapsed = (timeit.default_timer() - pad_start)
    total_time += pad_elapsed
    logger.info(f"{video_name} finish padding")

    # shift regions
    shift_start = timeit.default_timer()
    shift_image_direc = video_name + '-pack-images'
    if track_region:
        merged_regions_maps = pack_merged_regions_as_image(track_new_regions_dict, 
            req_new_regions_dict, merged_new_regions_contain_dict, 
            shift_image_direc, tmp_regions_direc, src_image_w, src_image_h)
    else:
        merged_regions_maps = pack_merged_regions_as_image(merged_new_regions_dict, 
            req_new_regions_dict, merged_new_regions_contain_dict, 
            shift_image_direc, tmp_regions_direc, src_image_w, src_image_h)
    shift_elapsed = (timeit.default_timer() - shift_start)
    total_time += shift_elapsed
    logger.info(f"{video_name} finish shifting")

    # perform detection
    infer_start = timeit.default_timer()
    frame_cnt = len(sorted(os.listdir(shift_image_direc)))
    pad_shift_results, rpn_results = server.perform_detection(shift_image_direc, high_resolution)
    infer_elapsed = (timeit.default_timer() - infer_start)
    total_time += infer_elapsed
    logger.info(f'{video_name} finish detection')

    # post-processing of detection results, merge boxes
    merge_start = timeit.default_timer()
    pad_shift_results = merge_boxes_in_results(pad_shift_results.regions_dict, 
        server.config.low_threshold, server.config.suppression_threshold)
    merge_elapsed = (timeit.default_timer() - merge_start)
    total_time += merge_elapsed

    if not cleanup:
        vis_pack_direc = video_name + '-vis-pack'
        pack_fnames = sorted([f for f in os.listdir(shift_image_direc) if "png" in f])
        draw_region_rectangle(shift_image_direc, pack_fnames, pad_shift_results.regions_dict, 
                vis_pack_direc, display_result=True)

    # restore results on these packed images
    restore_start = timeit.default_timer()

    if track_region:
        restored_pad_shift_results, restored_req_regions_id = \
            restore_track_regions_detection_box(
                pad_shift_results, merged_new_regions_dict, 
                track_new_regions_contain_dict, req_new_regions_dict, merged_new_regions_contain_dict,
                merged_regions_maps, src_image_w, src_image_h, iou_thresh=intersect_iou
            )
    else:
        restored_pad_shift_results, restored_req_regions_id = \
            restore_merged_regions_detection_box(
                pad_shift_results, merged_new_regions_dict, 
                req_new_regions_dict, merged_new_regions_contain_dict, 
                merged_regions_maps, src_image_w, src_image_h, iou_thresh=intersect_iou
            )
    restored_pad_shift_results = merge_boxes_in_results(restored_pad_shift_results.regions_dict, 
        server.config.low_threshold, server.config.suppression_threshold)
    restored_pad_shift_results.write(f"{video_name}-pack-txt")

    if not cleanup:
        # plot restored detection boxes
        visualize_image_direc = video_name + '-visualize-restore'
        visualize_image_tmp_direc = visualize_image_direc + '-tmp'
        fnames = sorted([f for f in os.listdir(region_image_direc) if "png" in f])
        draw_region_rectangle(region_image_direc, fnames, restored_pad_shift_results.regions_dict,
            visualize_image_tmp_direc, rec_side_width=4)

        # plot req regions
        restored_req_regions_dict = {}
        for req_id in restored_req_regions_id:
            fid = req_new_regions_dict[req_id].original_region.fid
            if fid not in restored_req_regions_dict.keys():
                restored_req_regions_dict[fid] = []
            restored_req_regions_dict[fid].append(req_new_regions_dict[req_id].original_region)

        draw_region_rectangle(visualize_image_tmp_direc, fnames, restored_req_regions_dict,
            visualize_image_direc, rec_side_width=4, rec_color=(0,255,0))
        shutil.rmtree(visualize_image_tmp_direc)

    # combine with low-quality results
    restored_pad_shift_results.combine_results(mpeg_regions_result, server.config.intersection_threshold)

    # merge detection box again
    restored_pad_shift_results = merge_boxes_in_results(restored_pad_shift_results.regions_dict, 
        server.config.low_threshold, server.config.suppression_threshold)
    restore_elapsed = (timeit.default_timer() - restore_start)
    total_time += restore_elapsed

    # visualize_regions(restored_pad_shift_results, region_image_direc)
    max_fid = max(list(restored_pad_shift_results.regions_dict.keys()))
    if gt_regions_dict:
        diff_image_direc = f'{video_name}-diff'
        two_results_diff(max_fid, gt_regions_dict, restored_pad_shift_results.regions_dict, 
            server.config.low_threshold, server.config.max_object_size, 
            server.config.prune_score, server.config.max_object_size, 
            server.config.objfilter_iou, vis_common=3,
            src_image_direc=region_image_direc, diff_image_direc=diff_image_direc,
            name1='gt', name2='packed')
    
    shutil.rmtree(shift_image_direc)
    logger.info(f"Padding time {pad_elapsed}, shifting time {shift_elapsed}, "
                f"infer time {infer_elapsed} for {frame_cnt} frames, "
                f"total elapsed time {total_time}")
    if out_cost_file:
        write_compute_cost(out_cost_file, video_name, frame_cnt, 
            pad_elapsed, shift_elapsed, infer_elapsed, total_time)
    
    # fill empty frames
    fnames = sorted([f for f in os.listdir(region_image_direc) if "png" in f])
    number_of_frames = len(list(fnames))
    restored_pad_shift_results.fill_gaps(number_of_frames)

    return restored_pad_shift_results, [0,0]