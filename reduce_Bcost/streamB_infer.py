import os
import logging
import argparse
from dds_utils import (visualize_regions, merge_boxes_in_results, Results, compute_regions_size, 
        Region)
from .merge_regions import (encode_regions, encode_regions_images, encode_batch_filtered_regions)
from .pad_regions import (pad_all_req_regions, pad_encoded_regions, 
        pad_filtered_regions_context, pad_filtered_regions_blank)
from .resize_regions import (resize_correspond_regions)
from .shift_regions import (pack_merged_regions_as_image, pack_filtered_padded_regions)
from .restore_results import (restore_detection_box, compress_restored_results, 
        restore_merged_regions_detection_box, restore_track_regions_detection_box)
from .streamB_utils import (get_percentile, two_results_diff, draw_region_rectangle, 
        filter_true_regions, write_compute_cost, track2req_region, merge_images_by_frame, 
        exclude_frame_regions)
from .region_similarity import (track_similar_region)
from .filter_regions import (filter_regions_dds)
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
        merge_iou=0.0, track_region=False, max_diff=0.0, max_frame_interval=0
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
            track_similar_region(merged_new_regions_dict, tmp_regions_direc, \
                max_diff, max_frame_interval)
        logger.info(f"{video_name} finish tracking")

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
        track_region_fname = video_name + '-track_region_dict'
        for cur_track_region in track_new_regions_dict.values():
            cur_track_region.write(track_region_fname)
        all_region_fname = video_name + '-all_region_dict'
        for cur_region in merged_new_regions_dict.values():
            cur_region.write(all_region_fname)
        contain_region_fname = video_name + '-contain_region_dict'
        with open(contain_region_fname, 'a') as f:
            f.write(str(track_new_regions_contain_dict))
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
                merged_regions_maps, src_image_w, src_image_h, 
                iou_thresh=intersect_iou
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

    # if not cleanup:
    #     # plot restored detection boxes
    #     visualize_image_direc = video_name + '-visualize-restore'
    #     visualize_image_tmp_direc = visualize_image_direc + '-tmp'
    #     fnames = sorted([f for f in os.listdir(region_image_direc) if "png" in f])
    #     draw_region_rectangle(region_image_direc, fnames, restored_pad_shift_results.regions_dict,
    #         visualize_image_tmp_direc, rec_side_width=4)

    #     # plot req regions
    #     restored_req_regions_dict = {}
    #     for req_id in restored_req_regions_id:
    #         fid = req_new_regions_dict[req_id].original_region.fid
    #         if fid not in restored_req_regions_dict.keys():
    #             restored_req_regions_dict[fid] = []
    #         restored_req_regions_dict[fid].append(req_new_regions_dict[req_id].original_region)

    #     draw_region_rectangle(visualize_image_tmp_direc, fnames, restored_req_regions_dict,
    #         visualize_image_direc, rec_side_width=4, rec_color=(0,255,0))
    #     shutil.rmtree(visualize_image_tmp_direc)

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


def analyze_combined_methods(
        server, video_name, region_image_direc, context_image_direc, high_resolution,
        req_regions_result, mpeg_regions_result, 
        context_padding_type, context_val, blank_padding_type, blank_val,
        cleanup=False, gt_regions_dict=None,
        out_cost_file=None, 
        intersect_iou=0.2,
        resize_type='fixed_ratio', area_upper_bound=0.0005, fixed_ratio=1, fixed_area=1, 
        merge_iou=0.0, max_diff=0.0, max_frame_interval=0
    ):

    logger = logging.getLogger("streamB-combined_methods")
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
    track_new_regions_dict, track_new_regions_contain_dict, new_regions_contained_track_dict = \
        track_similar_region(merged_new_regions_dict, tmp_regions_direc, \
            max_diff, max_frame_interval, inverse_contain=True)
    dds_req_regions_dict, fid_list = track2req_region(track_new_regions_dict, req_regions_result.regions_dict)
    import ipdb; ipdb.set_trace()
    logger.info(f"{video_name} finish tracking")
    pad_elapsed = (timeit.default_timer() - pad_start)
    total_time += pad_elapsed
    logger.info(f"{video_name} finish padding")

    # run DDS on frames containing track regions
    dds_image_direc = video_name + '-dds-images'
    merge_images_by_frame(dds_req_regions_dict, region_image_direc, context_image_direc, dds_image_direc)
    dds_results, dds_rpn_results = server.perform_detection(dds_image_direc, high_resolution)

    # shift regions
    shift_start = timeit.default_timer()
    shift_image_direc = video_name + '-pack-images'
    shift_new_regions_dict = exclude_frame_regions(merged_new_regions_dict, fid_list)
    merged_regions_maps = pack_merged_regions_as_image(shift_new_regions_dict, 
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

    # post-processing of detection results, use tracking to correct
    # restore results on these packed images
    restore_start = timeit.default_timer()
    restored_pad_shift_results, restored_req_regions_id = \
        restore_merged_regions_detection_box(
            pad_shift_results, merged_new_regions_dict, 
            req_new_regions_dict, merged_new_regions_contain_dict, 
            merged_regions_maps, src_image_w, src_image_h, iou_thresh=intersect_iou
        )
    
    restored_pad_shift_results = merge_boxes_in_results(restored_pad_shift_results.regions_dict, 
        server.config.low_threshold, server.config.suppression_threshold)
    # combine with dds results
    restored_pad_shift_results.combine_results(dds_results, server.config.intersection_threshold)
    restored_pad_shift_results.write(f"{video_name}-pack-txt")

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


def analyze_encoded_regions(
        server, video_name, req_regions_result, mpeg_regions_result,
        percent_list, qp_list, context_padding_type, context_val, blank_padding_type, blank_val,
        cleanup=False, out_cost_file=None, intersect_iou=0.2,
        merge_iou=0.0, track_region=False, max_diff=0.0, max_frame_interval=0
    ):

    logger = logging.getLogger("streamB-encode-pad-shift")
    handler = logging.NullHandler()
    logger.addHandler(handler)
    logger.info(f'{video_name} initialize')
    print(merge_iou, track_region)

    total_size = [0, 0]
    total_time = 0
    total_pad_time = 0
    total_shift_time = 0
    total_infer_time = 0
    all_restored_pad_shift_results = Results()
    fnames = sorted([f for f in os.listdir(server.config.high_images_path) if "png" in f])
    number_of_frames = len(list(fnames))
    start_bid = 0

    for i in range(0, number_of_frames, server.config.batch_size):
        start_fid = i
        end_fid = min(number_of_frames, i + server.config.batch_size)
        cur_req_regions_result = Results()
        for fid in range(start_fid, end_fid):
            if fid in req_regions_result.regions_dict.keys():
                for single_region in req_regions_result.regions_dict[fid]:
                    cur_req_regions_result.append(single_region)

        logger.info(f"Processing batch from {start_fid} to {end_fid}")

        # Encode regions with different qp
        encode_start = timeit.default_timer()
        merged_images_direc = f'{video_name}-{start_fid}-{end_fid}-merged'
        batch_size = encode_regions(server, logger, cur_req_regions_result, percent_list, qp_list, \
            server.config.batch_size, server.config.high_images_path, merged_images_direc)
        total_size[0] += batch_size[0]
        total_size[1] += batch_size[1]
        encode_elapsed = (timeit.default_timer() - encode_start)
        total_time += encode_elapsed
        logger.info(f'{video_name} finish encoding')

        # Pad regions
        pad_start = timeit.default_timer()
        req_new_regions_dict, merged_new_regions_dict, merged_new_regions_contain_dict, \
        context_padded_regions_direc, blank_padded_regions_direc, src_image_w, src_image_h = \
            pad_encoded_regions(
                cur_req_regions_result, video_name, merged_images_direc, 
                context_padding_type, context_val, blank_padding_type, blank_val, merge_iou, 
                track_region=track_region
            )
        pad_elapsed = (timeit.default_timer() - pad_start)
        total_time += pad_elapsed
        total_pad_time += pad_elapsed
        logger.info(f"{video_name} finish padding")
        shutil.rmtree(merged_images_direc)

        if track_region:
            track_new_regions_dict, track_new_regions_contain_dict = \
                track_similar_region(merged_new_regions_dict, context_padded_regions_direc, \
                    max_diff, max_frame_interval)
            logger.info(f"{video_name} finish tracking")
            shutil.rmtree(context_padded_regions_direc)

        # Shift regions
        shift_start = timeit.default_timer()
        shift_image_direc = video_name + '-pack-images'
        if track_region:
            merged_regions_maps = pack_merged_regions_as_image(track_new_regions_dict, 
                req_new_regions_dict, merged_new_regions_contain_dict, 
                shift_image_direc, blank_padded_regions_direc, src_image_w, src_image_h, start_bid=start_bid)
        else:
            merged_regions_maps = pack_merged_regions_as_image(merged_new_regions_dict, 
                req_new_regions_dict, merged_new_regions_contain_dict, 
                shift_image_direc, blank_padded_regions_direc, src_image_w, src_image_h, start_bid=start_bid)

        shift_elapsed = (timeit.default_timer() - shift_start)
        total_time += shift_elapsed
        total_shift_time += shift_elapsed
        frame_cnt = len(sorted(os.listdir(shift_image_direc)))
        start_bid += frame_cnt
        logger.info(f"{video_name} finish shifting")

        # Perform detection
        infer_start = timeit.default_timer()
        pad_shift_results, rpn_results = server.perform_detection(shift_image_direc, \
            server.config.high_resolution)
        infer_elapsed = (timeit.default_timer() - infer_start)
        total_time += infer_elapsed
        total_infer_time += infer_elapsed
        logger.info(f'{video_name} finish detection')

        # Post-processing of detection results, merge boxes
        merge_start = timeit.default_timer()
        pad_shift_results = merge_boxes_in_results(pad_shift_results.regions_dict, 
            server.config.low_threshold, server.config.suppression_threshold)
        merge_elapsed = (timeit.default_timer() - merge_start)
        total_time += merge_elapsed

        if not cleanup:
            vis_pack_direc = video_name + '-vis-pack'
            pack_fnames = sorted([f for f in os.listdir(shift_image_direc) if "png" in f])
            draw_region_rectangle(shift_image_direc, pack_fnames, pad_shift_results.regions_dict, 
                    vis_pack_direc, display_result=True, clean_save=False)

        # Restore results on these packed images
        restore_start = timeit.default_timer()
        if track_region:
            restored_pad_shift_results, restored_req_regions_id = \
                restore_track_regions_detection_box(
                    pad_shift_results, merged_new_regions_dict, 
                    track_new_regions_contain_dict, req_new_regions_dict, merged_new_regions_contain_dict,
                    merged_regions_maps, src_image_w, src_image_h, 
                    iou_thresh=intersect_iou
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

        if not cleanup:
            vis_restore_direc = video_name + '-vis-restore'
            restore_fnames = []
            for cur_fid in range(start_fid, end_fid):
                restore_fnames.append(f"{str(cur_fid).zfill(10)}.png")
            draw_region_rectangle(server.config.high_images_path, restore_fnames, \
                restored_pad_shift_results.regions_dict, vis_restore_direc, \
                display_result=True, clean_save=False)

        all_restored_pad_shift_results.combine_results(restored_pad_shift_results, 
            server.config.intersection_threshold)
        restore_elapsed = (timeit.default_timer() - restore_start)
        total_time += restore_elapsed
        shutil.rmtree(shift_image_direc)

    all_restored_pad_shift_results.write(f"{video_name}-pack-txt")
    # combine with low-quality results
    all_restored_pad_shift_results.combine_results(mpeg_regions_result, server.config.intersection_threshold)

    # merge detection box again
    all_restored_pad_shift_results = merge_boxes_in_results(all_restored_pad_shift_results.regions_dict, 
        server.config.low_threshold, server.config.suppression_threshold)

    logger.info(f"Padding time {total_pad_time}, shifting time {total_shift_time}, "
                f"infer time {total_infer_time} for {frame_cnt} frames, "
                f"total elapsed time {total_time}")
    if out_cost_file:
        write_compute_cost(out_cost_file, video_name, frame_cnt, 
            total_pad_time, total_shift_time, total_infer_time, total_time)
    
    # Fill empty frames
    all_restored_pad_shift_results.fill_gaps(number_of_frames)

    return all_restored_pad_shift_results, total_size


def analyze_dds_filtered(
        server, video_name, req_regions_result, mpeg_regions_result,
        context_padding_type, context_val, blank_padding_type, blank_val,
        resize_method='no', cleanup=False, out_cost_file=None, intersect_iou=0.2, merge_iou=0.0, 
        filter_by_dds=False,
    ):

    logger = logging.getLogger("streamB-dds-filtered")
    handler = logging.NullHandler()
    logger.addHandler(handler)
    logger.info(f'{video_name} initialize')

    total_size = [0, 0]
    total_time = 0
    total_pad_time = 0
    total_shift_time = 0
    total_infer_time = 0
    total_frame_cnt = 0
    all_restored_results = Results()
    dnn_output_results = Results()
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

            # Filter req_region in the batch
            cur_req_regions_result = filter_regions_dds(
                dds_results, cur_req_regions_result, start_fid,f'{base_images_direc}-cropped',
                match_thresh=0.9, filter_iou=0.0, 
                confid_thresh=server.config.prune_score, max_object_size=server.config.max_object_size
            )

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
        # Then resize small regions
        resize_correspond_regions(
            req_new_regions_dict, merged_new_regions_dict, merged_new_regions_contain_dict, 
            mpeg_regions_result, resize_method
        )
        # Last do blank paddding
        padded_regions_direc, src_image_w, src_image_h, merged_new_regions_dict = pad_filtered_regions_blank(
            merged_new_regions_dict, batch_images_direc, 
            f'{merged_images_direc}-cropped', blank_padding_type, blank_val, sort_context_region=True
        )
        pad_elapsed = (timeit.default_timer() - pad_start)
        total_pad_time += pad_elapsed
        logger.info(f"{video_name} finish padding")

        # Shift padded regions
        shift_start = timeit.default_timer()
        shift_images_direc = f'{batch_images_direc}-pack'
        merged_regions_maps = pack_filtered_padded_regions(
            req_new_regions_dict, merged_new_regions_dict, merged_new_regions_contain_dict, 
            shift_images_direc, padded_regions_direc, src_image_w, src_image_h, start_bid=start_bid, 
            merged_images_direc=f'{merged_images_direc}-cropped'
        )

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
        # restored_pad_shift_results = merge_boxes_in_results(restored_pad_shift_results.regions_dict, 
        #     server.config.low_threshold, server.config.suppression_threshold)

        if not cleanup:
            vis_pack_direc = f'{video_name}-vis-pack'
            pack_fnames = sorted([f for f in os.listdir(shift_images_direc) if "png" in f])
            draw_region_rectangle(shift_images_direc, pack_fnames, pad_shift_results.regions_dict, 
                vis_pack_direc, display_result=True, drop_no_rect=True, clean_save=False)
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
        if filter_by_dds:
            # Combine with results on DDS image
            all_restored_results.combine_results(dds_results, 
                server.config.intersection_threshold)
        restore_elapsed = (timeit.default_timer() - restore_start)
        total_time += restore_elapsed

        # Remove related directory
        shutil.rmtree(shift_images_direc)
        shutil.rmtree(f'{merged_images_direc}-cropped')
        shutil.rmtree(f'{base_images_direc}-cropped')
        if filter_by_dds:
            shutil.rmtree(f'{dds_images_direc}-cropped')

    dnn_output_results.write(f"{video_name}-dnn-txt")
    all_restored_results.write(f"{video_name}-pack-txt")
    # combine with low-quality results
    all_restored_results.combine_results(mpeg_regions_result, server.config.intersection_threshold)
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


