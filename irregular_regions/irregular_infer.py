from .prepare_regions import (read_object_regions, get_largest_bbox, wrap_bboxes)
from .pack_regions import (pack_bboxes)
from .restore_results import (restore_detection_boxes)
from dds_utils import (read_results_dict, Results, merge_boxes_in_results)
from reduce_Bcost.streamB_utils import (draw_region_rectangle, write_compute_cost)
import os
import logging
import argparse
import timeit
import shutil


def analyze_only_objects(
        server, video_name, region_path, region_enlarge_ratio, src_images_direc, 
        cleanup, out_cost_file=None):
    logger = logging.getLogger("streamB-dds-filtered")
    logger.addHandler(logging.NullHandler())
    logger.warning(f'{video_name} initialize')

    total_size = [0, 0]
    total_time = 0
    total_pad_time = 0
    total_shift_time = 0
    total_infer_time = 0
    total_frame_cnt = 0
    all_restored_results = Results()
    dnn_output_results = Results()
    fnames = sorted([f for f in os.listdir(src_images_direc) if "png" in f])
    number_of_frames = len(list(fnames))
    start_bid = 0
    total_start = timeit.default_timer()

    regions_dict = read_results_dict(region_path)
    for i in range(0, number_of_frames, server.config.batch_size):
        start_fid = i
        end_fid = min(number_of_frames, i + server.config.batch_size)
        logger.info(f"Processing batch from {start_fid} to {end_fid}")
        batch_images_direc = f'{video_name}-{start_fid}-{end_fid}'
        batch_fnames = [f"{str(f).zfill(10)}.png" for f in range(start_fid, end_fid)]

        pad_start = timeit.default_timer()
        region_images_direc = f"{batch_images_direc}-region_images"
        region_map_dict = read_object_regions(regions_dict, region_enlarge_ratio, batch_fnames, \
            src_images_direc, region_images_direc, server.config.max_object_size)
        all_bboxes, region_id_map_dict, region_area_dict, src_image_w, src_image_h = \
            get_largest_bbox(region_map_dict, region_images_direc)
        all_new_regions_dict = wrap_bboxes(all_bboxes)
        pad_elapsed = (timeit.default_timer() - pad_start)
        total_pad_time += pad_elapsed
        logger.info("Finish getting regions")

        shift_start = timeit.default_timer()
        pack_images_direc = f"{batch_images_direc}-pack_images"
        pack_regions_map = pack_bboxes(all_new_regions_dict, region_id_map_dict, region_images_direc, \
            pack_images_direc, src_image_w, src_image_h, start_bid)
        shift_elapsed = (timeit.default_timer() - shift_start)
        total_shift_time += shift_elapsed
        frame_cnt = len(sorted(os.listdir(pack_images_direc)))
        start_bid += frame_cnt
        total_frame_cnt += frame_cnt
        logger.info("Finish packing regions")

        # Perform detection
        infer_start = timeit.default_timer()
        pack_results, rpn_results, _ = server.perform_detection(pack_images_direc, \
            server.config.high_resolution)
        infer_elapsed = (timeit.default_timer() - infer_start)
        total_infer_time += infer_elapsed
        logger.info("Finish performing detections")
        dnn_output_results.combine_results(pack_results, server.config.intersection_threshold)

        if not cleanup:
            vis_pack_direc = f"{video_name}-vis_pack"
            pack_fnames = sorted([f for f in os.listdir(pack_images_direc) if "png" in f])
            draw_region_rectangle(pack_images_direc, pack_fnames, pack_results.regions_dict, 
                vis_pack_direc, display_result=True, drop_no_rect=False, clean_save=False)
        
        # Restore results
        restored_pack_results, restored_region_ids = restore_detection_boxes(
            pack_results, all_new_regions_dict, region_area_dict, pack_regions_map, 
            src_image_w, src_image_h)
        all_restored_results.combine_results(restored_pack_results, 
            server.config.intersection_threshold)
        
        if not cleanup:
            vis_restore_direc = f"{video_name}-vis_restore"
            draw_region_rectangle(server.config.high_images_path, batch_fnames, \
                restored_pack_results.regions_dict, vis_restore_direc, \
                display_result=True, clean_save=False, drop_no_rect=True)
        
        # Remove related directory
        shutil.rmtree(pack_images_direc)
        import ipdb; ipdb.set_trace()

    dnn_output_results.write(f"{video_name}-dnn")
    all_restored_results = merge_boxes_in_results(all_restored_results.regions_dict, 
        server.config.low_threshold, server.config.suppression_threshold)
    all_restored_results.write(f"{video_name}")

    total_time = timeit.default_timer() - total_start
    logger.info(f"Padding time {total_pad_time}, shifting time {total_shift_time}, "
                f"infer time {total_infer_time} for {total_frame_cnt} frames, "
                f"total elapsed time {total_time}")
    if out_cost_file:
        write_compute_cost(out_cost_file, video_name, total_frame_cnt, 
            total_pad_time, total_shift_time, total_infer_time, total_time)
    
    # Fill empty frames
    all_restored_results.fill_gaps(number_of_frames)

    return all_restored_results, total_size