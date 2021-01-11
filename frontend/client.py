import logging
import os
import shutil
import requests
import json
from dds_utils import (Results, read_results_dict, cleanup, Region,
                       compute_regions_size, extract_images_from_video,
                       merge_boxes_in_results, write_compute_cost, calc_iou)
from reduce_Acost.merge_low_frames import (merge_frames, restore_merged_results, 
                                merge_frame_change_res, restore_frame_change_res)
from reduce_Acost.crop_merge_frames import (crop_merge_frames_part)
from reduce_Bcost.streamB_utils import (draw_region_rectangle, filter_true_regions)
from reduce_Bcost.filter_regions import (get_filtered_req_regions)
import ipdb
import timeit

class Client:
    """The client of the DDS protocol
       sends images in low resolution and waits for
       further instructions from the server. And finally receives results
       Note: All frame ranges are half open ranges"""

    def __init__(self, hname, config, server_handle=None):
        if hname:
            self.hname = hname
            self.session = requests.Session()
        else:
            self.server = server_handle
        self.config = config

        self.logger = logging.getLogger("client")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.logger.info(f"Client initialized")

    def analyze_video_mpeg(self, video_name, raw_images_path, enforce_iframes, 
            out_cost_file=None):
        number_of_frames = len(
            [f for f in os.listdir(raw_images_path) if ".png" in f])

        final_results = Results()
        final_rpn_results = Results()
        total_size = 0
        total_infer_time = 0
        total_time = 0
        
        for i in range(0, number_of_frames, self.config.batch_size):
            start_frame = i
            end_frame = min(number_of_frames, i + self.config.batch_size)

            batch_fnames = sorted([f"{str(idx).zfill(10)}.png"
                                   for idx in range(start_frame, end_frame)])

            req_regions = Results()
            for fid in range(start_frame, end_frame):
                req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.low_resolution))
            batch_video_size, _ = compute_regions_size(
                req_regions, f"{video_name}-base-phase", raw_images_path,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            self.logger.info(f"{batch_video_size / 1024}KB sent "
                             f"in base phase using {self.config.low_qp}QP")
            extract_images_from_video(f"{video_name}-base-phase-cropped",
                                      req_regions)
            infer_start = timeit.default_timer()
            results, rpn_results, _ = (
                self.server.perform_detection(
                    f"{video_name}-base-phase-cropped",
                    self.config.low_resolution, batch_fnames))
            infer_elapsed = (timeit.default_timer() - infer_start)
            total_infer_time += infer_elapsed

            self.logger.info(f"Detection {len(results)} regions for "
                             f"batch {start_frame} to {end_frame} with a "
                             f"total size of {batch_video_size / 1024}KB")
            combine_start = timeit.default_timer()
            final_results.combine_results(
                results, self.config.intersection_threshold)
            combine_elapsed = (timeit.default_timer() - combine_start)
            total_time += combine_elapsed
            final_rpn_results.combine_results(
                rpn_results, self.config.intersection_threshold)

            # Remove encoded video manually
            shutil.rmtree(f"{video_name}-base-phase-cropped")
            total_size += batch_video_size
        
        combine_start = timeit.default_timer()
        final_results = merge_boxes_in_results(
            final_results.regions_dict, 
            self.config.low_threshold, self.config.suppression_threshold)
        combine_elapsed = (timeit.default_timer() - combine_start)
        total_time += combine_elapsed
        total_time += total_infer_time
        # final_rpn_results = merge_boxes_in_results(
        #     final_rpn_results.regions_dict, 
        #     -1, self.config.suppression_threshold)
        final_results.fill_gaps(number_of_frames)

        # Add RPN regions
        final_results.combine_results(
            final_rpn_results, self.config.intersection_threshold)

        final_results.write(video_name)
        self.logger.info(f"Infer time {total_infer_time} for "
                        f"{number_of_frames} frames, "
                        f"total elapsed time {total_time}")
        if out_cost_file:
            write_compute_cost(out_cost_file, video_name, number_of_frames, 
                0, 0, total_infer_time, total_time)

        return final_results, [total_size, 0]
    

    def analyze_video_reduced_mpeg(self, video_name, raw_images_path, enforce_iframes, 
            whole_res, merged_frame_res, out_cost_file=None):
        number_of_frames = len(
            [f for f in os.listdir(raw_images_path) if ".png" in f])

        final_results = Results()
        final_rpn_results = Results()
        total_size = 0
        total_infer_time = 0
        total_time = 0
        total_start = timeit.default_timer()
        total_frame_cnt = 0
        
        for i in range(0, number_of_frames, self.config.batch_size):
            start_frame = i
            end_frame = min(number_of_frames, i + self.config.batch_size)
            self.logger.info(f"Processing batch from {start_frame} to {end_frame}")

            batch_fnames = sorted([f"{str(idx).zfill(10)}.png"
                                   for idx in range(start_frame, end_frame)])

            # Get frames encoded with low_resolution
            req_regions = Results()
            for fid in range(start_frame, end_frame):
                req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2, whole_res))
            batch_video_size, _ = compute_regions_size(
                req_regions, f"{video_name}-base-phase", raw_images_path,
                whole_res, self.config.low_qp,
                enforce_iframes, True)
            self.logger.info(f"{batch_video_size / 1024}KB sent "
                             f"in base phase using {self.config.low_qp}QP")
            extract_images_from_video(f"{video_name}-base-phase-cropped",
                                      req_regions)
            base_images_direc = f"{video_name}-base-phase-cropped"

            # Merge frames into a large one
            merged_frame_direc = f"{video_name}_merged_direc"
            merged_maps = merge_frame_change_res(base_images_direc, req_regions.regions, \
                whole_res, merged_frame_res, whole_res, merged_frame_direc)
            merged_frame_fnames = sorted([f for f in os.listdir(merged_frame_direc) if "png" in f])
            total_frame_cnt += len(merged_frame_fnames)
            merged_results, merged_rpn_results, merged_all_rpn = \
                self.server.perform_detection(merged_frame_direc, 
                    whole_res, merged_frame_fnames)
            
            # Restore results
            restored_merged_results = restore_frame_change_res(merged_results, merged_maps,\
                # min_area_thresh=self.config.size_obj / ((whole_res/large_object_res)**2) 
            )
            final_results.combine_results(restored_merged_results, \
                self.config.intersection_threshold)
            vis_merged_res_direc = f"{video_name}_vis_merged_res"
            draw_region_rectangle(merged_frame_direc, merged_frame_fnames, \
                merged_results.regions_dict, vis_merged_res_direc)
            vis_restored_results_direc = f"{video_name}_vis_restored_res"
            draw_region_rectangle(base_images_direc, batch_fnames, \
                restored_merged_results.regions_dict, vis_restored_results_direc, clean_save=False)
            
            # Select from all rpn
            restored_merged_all_rpn = restore_frame_change_res(merged_all_rpn, merged_maps, \
                # min_area_thresh=self.config.size_obj / ((whole_res/large_object_res)**2) 
            )
            selected_restored_rpn = self.server.perform_non_max_suppression(
                restored_merged_all_rpn)
            final_rpn_results.combine_results(selected_restored_rpn, \
                self.config.intersection_threshold)
            # vis_selected_restored_rpn = f"{video_name}-vis_selected_restored_rpn"
            # draw_region_rectangle(base_images_direc, batch_fnames, \
            #     selected_restored_rpn.regions_dict, vis_selected_restored_rpn)
            
            shutil.rmtree(merged_frame_direc)
            import ipdb; ipdb.set_trace()

            # Remove encoded video manually
            shutil.rmtree(base_images_direc)
            total_size += batch_video_size
        
        final_results = merge_boxes_in_results(
            final_results.regions_dict, 
            self.config.low_threshold, self.config.suppression_threshold)
        
        # final_rpn_results = merge_boxes_in_results(
        #     final_rpn_results.regions_dict, 
        #     -1, self.config.suppression_threshold)
        final_results.fill_gaps(number_of_frames)

        # Add RPN regions
        final_results.combine_results(
            final_rpn_results, self.config.intersection_threshold)

        final_results.write(video_name)
        total_time = (timeit.default_timer() - total_start)
        self.logger.info(f"Infer time {total_infer_time} for "
                        f"{total_frame_cnt} frames, "
                        f"total elapsed time {total_time}")
        if out_cost_file:
            write_compute_cost(out_cost_file, video_name, total_frame_cnt, 
                0, 0, total_infer_time, total_time)

        return final_results, [total_size, 0]
        
    def analyze_video_merged_mpeg(self, video_name, raw_images_path, enforce_iframes, 
            whole_res, large_object_res, small_object_res,
            out_cost_file=None, iou_thresh=0.3, max_area_thresh=0.3):
        number_of_frames = len(
            [f for f in os.listdir(raw_images_path) if ".png" in f])

        final_results = Results()
        final_rpn_results = Results()
        total_size = 0
        total_infer_time = 0
        total_time = 0
        total_start = timeit.default_timer()
        total_frame_cnt = 0
        
        for i in range(0, number_of_frames, self.config.batch_size):
            start_frame = i
            end_frame = min(number_of_frames, i + self.config.batch_size)
            self.logger.info(f"Processing batch from {start_frame} to {end_frame}")

            batch_fnames = sorted([f"{str(idx).zfill(10)}.png"
                                   for idx in range(start_frame, end_frame)])

            # Get frames encoded with low_resolution
            req_regions = Results()
            for fid in range(start_frame, end_frame):
                req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2, whole_res))
            batch_video_size, _ = compute_regions_size(
                req_regions, f"{video_name}-base-phase", raw_images_path,
                whole_res, self.config.low_qp,
                enforce_iframes, True)
            self.logger.info(f"{batch_video_size / 1024}KB sent "
                             f"in base phase using {self.config.low_qp}QP")
            extract_images_from_video(f"{video_name}-base-phase-cropped",
                                      req_regions)
            base_images_direc = f"{video_name}-base-phase-cropped"

            # Get original results on the first frame
            first_fname = [f"{str(start_frame).zfill(10)}.png"]
            total_frame_cnt += len(first_fname)
            ori_first_results, ori_first_rpn_results, _ = \
                self.server.perform_detection(base_images_direc, 
                    whole_res, first_fname)
            # Get repeated results on the first frame
            repeat_first_times = int((whole_res // large_object_res) ** 2)
            first_region = Region(start_frame, 0, 0, 1, 1, 1.0, 2, whole_res)
            first_region_list = []
            for i in range(repeat_first_times):
                first_region_list.append(first_region)
            repeat_first_direc = f"{video_name}-first_merged"
            first_frame_merged_maps = merge_frame_change_res(
                base_images_direc, first_region_list, whole_res, large_object_res, whole_res,
                repeat_first_direc
            )
            repeat_first_fname = sorted([f for f in os.listdir(repeat_first_direc) if "png" in f])
            total_frame_cnt += len(repeat_first_fname)
            repeat_first_results, repeat_first_rpn_result, _ = \
                self.server.perform_detection(repeat_first_direc, 
                    whole_res, repeat_first_fname)
            # vis_merged_res_direc = f"{video_name}-vis_merged_res"
            # vis_merged_rpn_direc = f"{video_name}-vis_merged_rpn"
            # draw_region_rectangle(repeat_first_direc, repeat_first_fname, \
            #     repeat_first_results.regions_dict, vis_merged_res_direc)
            # draw_region_rectangle(repeat_first_direc, repeat_first_fname, \
            #     repeat_first_rpn_result.regions_dict, vis_merged_rpn_direc)
            shutil.rmtree(repeat_first_direc)

            # Restore repeated results
            first_frame_merged_maps[0] = [first_frame_merged_maps[0][0]]
            restored_repeat_first_results = restore_frame_change_res(
                repeat_first_results, first_frame_merged_maps)
            restored_repeat_first_rpn_results = restore_frame_change_res(
                repeat_first_rpn_result, first_frame_merged_maps)
            # vis_res_direc = f"{video_name}-vis_res"
            # vis_rpn_direc = f"{video_name}-vis_rpn"
            # draw_region_rectangle(base_images_direc, first_fname, \
            #     restored_repeat_first_results.regions_dict, vis_res_direc)
            # draw_region_rectangle(base_images_direc, first_fname, \
            #     restored_repeat_first_rpn_results.regions_dict, vis_rpn_direc)

            # Put first frame results into final
            final_results.combine_results(ori_first_results, \
                self.config.intersection_threshold)
            final_rpn_results.combine_results(ori_first_rpn_results, \
                self.config.intersection_threshold)

            # Check if some regions are not detected in merged first frame
            ori_all_results = Results()
            ori_all_results.combine_results(ori_first_results, self.config.intersection_threshold)
            ori_all_results.combine_results(ori_first_rpn_results, self.config.intersection_threshold)
            repeat_all_results = Results()
            repeat_all_results.combine_results(restored_repeat_first_results, 
                self.config.intersection_threshold)
            repeat_all_results.combine_results(restored_repeat_first_rpn_results, 
                self.config.intersection_threshold)
            missed_results = Results()
            for fid, ori_region_list in ori_all_results.regions_dict.items():
                for ori_region in ori_region_list:
                    if ori_region.w * ori_region.h > max_area_thresh:
                        continue
                    find_match = False
                    if fid not in repeat_all_results.regions_dict.keys():
                        missed_results.append(ori_region)
                        continue
                    for repeat_region in repeat_all_results.regions_dict[fid]:
                        if calc_iou(ori_region, repeat_region) > iou_thresh:
                            find_match = True
                            break
                    if not find_match:
                        ori_region.label = "object"
                        missed_results.append(ori_region)
            
            # vis_miss_direc = f"{video_name}-vis_miss"
            # draw_region_rectangle(base_images_direc, first_fname, 
            #     missed_results.regions_dict, vis_miss_direc)

            # Get those merged, enlarged regions in each remaining frame
            missed_results = merge_boxes_in_results(missed_results.regions_dict, -1, 0)
            remain_small_regions_list = []
            for fid, regions_list in missed_results.regions_dict.items():
                for single_region in regions_list:
                    single_region.enlarge(self.config.enlarge_first_rpn)
                    for region_fid in range(start_frame+1, end_frame):
                        copy_single_region = single_region.copy()
                        copy_single_region.fid = region_fid
                        remain_small_regions_list.append(copy_single_region)
            
            # First run on entire frame
            remain_whole_region_list = []
            remain_fnames = []
            for region_fid in range(start_frame+1, end_frame):
                remain_whole_region_list.append(
                    Region(region_fid, 0, 0, 1, 1, 1.0, 2, whole_res))
                remain_fnames.append(f"{str(region_fid).zfill(10)}.png")
            remain_whole_direc = f"{video_name}_whole_direc"
            merged_whole_maps = merge_frame_change_res(base_images_direc, remain_whole_region_list, \
                whole_res, large_object_res, whole_res, remain_whole_direc)
            remain_whole_fnames = sorted([f for f in os.listdir(remain_whole_direc) if "png" in f])
            total_frame_cnt += len(remain_whole_fnames)
            whole_results, whole_rpn_results, whole_all_rpn = \
                self.server.perform_detection(remain_whole_direc, 
                    whole_res, remain_whole_fnames)
            vis_merged_whole_res = f"{video_name}-vis_whole_merged_res"
            draw_region_rectangle(remain_whole_direc, remain_whole_fnames, \
                whole_results.regions_dict, vis_merged_whole_res, display_result=True)
            vis_merged_whole_all_rpn = f"{video_name}-vis_whole_merged_all_rpn"
            draw_region_rectangle(remain_whole_direc, remain_whole_fnames, \
                whole_all_rpn.regions_dict, vis_merged_whole_all_rpn)
            restored_whole_results = restore_frame_change_res(whole_results, merged_whole_maps,\
                # min_area_thresh=self.config.size_obj / ((whole_res/large_object_res)**2) 
            )
            restored_whole_rpn_results = restore_frame_change_res(whole_rpn_results, merged_whole_maps, \
                # min_area_thresh=self.config.size_obj / ((whole_res/large_object_res)**2) 
            )
            final_results.combine_results(restored_whole_results, \
                self.config.intersection_threshold)
            final_rpn_results.combine_results(restored_whole_rpn_results, \
                self.config.intersection_threshold)
            vis_whole_res = f"{video_name}-vis_whole_res"
            vis_whole_rpn = f"{video_name}-vis_whole_rpn"
            draw_region_rectangle(base_images_direc, remain_fnames, \
                restored_whole_results.regions_dict, vis_whole_res, display_result=True)
            draw_region_rectangle(base_images_direc, remain_fnames, \
                restored_whole_rpn_results.regions_dict, vis_whole_rpn)
            shutil.rmtree(remain_whole_direc)

            # Then run on small regions
            remain_small_direc = f"{video_name}_small_direc"
            merged_small_maps = merge_frame_change_res(base_images_direc, remain_small_regions_list, \
                whole_res, small_object_res, whole_res, remain_small_direc)
            remain_small_fnames = sorted([f for f in os.listdir(remain_small_direc) if "png" in f])
            total_frame_cnt += len(remain_small_fnames)
            small_results, small_rpn_results, small_all_rpn  = \
                self.server.perform_detection(remain_small_direc, \
                    whole_res, remain_small_fnames)
            # vis_merged_small_res = f"{video_name}-vis_small_merged_res"
            # draw_region_rectangle(remain_small_direc, remain_small_fnames, \
            #     small_results.regions_dict, vis_merged_small_res, display_result=True)
            # vis_merged_small_all_rpn = f"{video_name}-vis_small_merged_all_rpn"
            # draw_region_rectangle(remain_small_direc, remain_small_fnames, \
            #     small_all_rpn.regions_dict, vis_merged_small_all_rpn)
            restored_small_results = restore_frame_change_res(small_results, merged_small_maps, \
                max_area_thresh=self.config.size_obj)
            restored_small_rpn_results = restore_frame_change_res(small_rpn_results, merged_small_maps, \
                max_area_thresh=self.config.size_obj)
            final_results.combine_results(restored_small_results, \
                self.config.intersection_threshold)
            final_rpn_results.combine_results(restored_small_rpn_results, \
                self.config.intersection_threshold)
            # vis_small_res = f"{video_name}-vis_small_res"
            # vis_small_rpn = f"{video_name}-vis_small_rpn"
            # draw_region_rectangle(base_images_direc, remain_fnames, \
            #     restored_small_results.regions_dict, vis_small_res, display_result=True)
            # draw_region_rectangle(base_images_direc, remain_fnames, \
            #     restored_small_rpn_results.regions_dict, vis_small_rpn)
            shutil.rmtree(remain_small_direc)

            # Remove encoded video manually
            shutil.rmtree(base_images_direc)
            total_size += batch_video_size
            import ipdb; ipdb.set_trace()
        
        final_results = merge_boxes_in_results(
            final_results.regions_dict, 
            self.config.low_threshold, self.config.suppression_threshold)
        
        # final_rpn_results = merge_boxes_in_results(
        #     final_rpn_results.regions_dict, 
        #     -1, self.config.suppression_threshold)
        final_results.fill_gaps(number_of_frames)

        # Add RPN regions
        final_results.combine_results(
            final_rpn_results, self.config.intersection_threshold)

        final_results.write(video_name)
        total_time = (timeit.default_timer() - total_start)
        self.logger.info(f"Infer time {total_infer_time} for "
                        f"{total_frame_cnt} frames, "
                        f"total elapsed time {total_time}")
        if out_cost_file:
            write_compute_cost(out_cost_file, video_name, total_frame_cnt, 
                0, 0, total_infer_time, total_time)

        return final_results, [total_size, 0]

    def analyze_video_emulate(self, video_name, high_images_path,
                              enforce_iframes, low_results_path=None,
                              debug_mode=False, out_cost_file=None):
        final_results = Results()
        low_phase_results = Results()
        high_phase_results = Results()
        total_req_regions = Results()
        filtered_total_req_regions = Results()
        total_infer_time = 0
        total_time = 0

        number_of_frames = len(
            [x for x in os.listdir(high_images_path) if "png" in x])

        low_results_dict = None
        if low_results_path:
            low_results_dict = read_results_dict(low_results_path)

        total_size = [0, 0]
        total_regions_count = 0
        for i in range(0, number_of_frames, self.config.batch_size):
            start_fid = i
            end_fid = min(number_of_frames, i + self.config.batch_size)
            self.logger.info(f"Processing batch from {start_fid} to {end_fid}")

            # Encode frames in batch and get size
            # Make temporary frames to downsize complete frames
            base_req_regions = Results()
            for fid in range(start_fid, end_fid):
                base_req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.high_resolution))
            encoded_batch_video_size, batch_pixel_size = compute_regions_size(
                base_req_regions, f"{video_name}-base-phase", high_images_path,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            self.logger.info(f"Sent {encoded_batch_video_size / 1024} "
                             f"in base phase")
            total_size[0] += encoded_batch_video_size

            # Low resolution phase
            low_images_path = f"{video_name}-base-phase-cropped"
            low_start = timeit.default_timer()
            r1, req_regions = self.server.simulate_low_query(
                start_fid, end_fid, low_images_path, low_results_dict, False,
                self.config.rpn_enlarge_ratio)
            total_regions_count += len(req_regions)
            
            final_results.combine_results(
                r1, self.config.intersection_threshold)
            low_elapsed = (timeit.default_timer() - low_start)
            total_time += low_elapsed
            low_phase_results.combine_results(
                r1, self.config.intersection_threshold)           

            # High resolution phase
            if len(req_regions) > 0:
                # Add new batch req_regions into total_req_regions
                total_req_regions.combine_results(
                    req_regions, self.config.intersection_threshold)
                # Crop, compress and get size
                regions_size, _ = compute_regions_size(
                    req_regions, video_name, high_images_path,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True)
                self.logger.info(f"Sent {len(req_regions)} regions which have "
                                 f"{regions_size / 1024}KB in second phase "
                                 f"using {self.config.high_qp}")
                total_size[1] += regions_size

                # High resolution phase every three filter
                r2, infer_elapsed = self.server.emulate_high_query(
                    video_name, low_images_path, req_regions)
                total_infer_time += infer_elapsed
                self.logger.info(f"Got {len(r2)} results in second phase "
                                 f"of batch")
                
                # Filter req_regions in this batch by r2
                filtered_req_regions_result, dds_results = \
                    get_filtered_req_regions(r2, req_regions, start_fid, low_images_path)
                filtered_total_req_regions.combine_results(
                    filtered_req_regions_result, self.config.intersection_threshold)

                high_phase_results.combine_results(
                    r2, self.config.intersection_threshold)
                combine_start = timeit.default_timer()
                final_results.combine_results(
                    r2, self.config.intersection_threshold)
                combine_elapsed = (timeit.default_timer() - combine_start)
                total_time += combine_elapsed

            # Cleanup for the next batch
            cleanup(video_name, debug_mode, start_fid, end_fid)

        self.logger.info(f"Got {len(low_phase_results)} unique results "
                         f"in base phase")
        self.logger.info(f"Got {len(high_phase_results)} positive "
                         f"identifications out of {total_regions_count} "
                         f"requests in second phase")

        # Fill gaps in results
        final_results.fill_gaps(number_of_frames)

        # Write results
        final_results.write(f"{video_name}")

        self.logger.info(f"Writing results for {video_name}")
        self.logger.info(f"{len(final_results)} objects detected "
                         f"and {total_size[1]} total size "
                         f"of regions sent in high resolution")

        rdict = read_results_dict(f"{video_name}")
        combine_start = timeit.default_timer()
        final_results = merge_boxes_in_results(rdict, 
            self.config.low_threshold, self.config.suppression_threshold)
        combine_elapsed = (timeit.default_timer() - combine_start)
        total_time += combine_elapsed
        total_time += total_infer_time

        # total_req_regions = merge_boxes_in_results(total_req_regions.regions_dict, 
        #     -1, self.config.suppression_threshold)
        # high_phase_results = merge_boxes_in_results(high_phase_results.regions_dict, 
        #     self.config.low_threshold, self.config.suppression_threshold)

        final_results.fill_gaps(number_of_frames)
        final_results.write(f"{video_name}")
        total_req_regions.write(f"{video_name}-req_regions")
        filtered_total_req_regions.write(f"{video_name}-filtered_req_regions")
        high_phase_results.write(f"{video_name}-high_phase_results")

        
        self.logger.info(f"Infer time {total_infer_time} for "
                        f"{number_of_frames} frames, "
                        f"total elapsed time {total_time}")
        if out_cost_file:
            write_compute_cost(out_cost_file, video_name, number_of_frames, 
                0, 0, total_infer_time, total_time)

        return final_results, total_size

    def init_server(self, nframes):
        params = dict(self.config.__dict__)
        params.update({"nframes": nframes})
        response = self.session.post(
            "http://" + self.hname + "/init", params=params)
        if response.status_code != 200:
            self.logger.fatal("Could not initialize server")
            # Need to add exception handling
            exit()

    def get_first_phase_results(self, vid_name):
        encoded_vid_path = os.path.join(
            vid_name + "-base-phase-cropped", "temp.mp4")
        video_to_send = {"media": open(encoded_vid_path, "rb")}
        response = self.session.post(
            "http://" + self.hname + "/low", files=video_to_send)
        response_json = json.loads(response.text)

        results = Results()
        for region in response_json["results"]:
            results.append(Region.convert_from_server_response(
                region, self.config.low_resolution, "low-res"))
        rpn = Results()
        for region in response_json["req_regions"]:
            rpn.append(Region.convert_from_server_response(
                region, self.config.low_resolution, "low-res"))

        return results, rpn

    def get_second_phase_results(self, vid_name):
        encoded_vid_path = os.path.join(vid_name + "-cropped", "temp.mp4")
        video_to_send = {"media": open(encoded_vid_path, "rb")}
        response = self.session.post(
            "http://" + self.hname + "/high", files=video_to_send)
        response_json = json.loads(response.text)

        results = Results()
        for region in response_json["results"]:
            results.append(Region.convert_from_server_response(
                region, self.config.high_resolution, "high-res"))

        return results

    def analyze_video(
            self, vid_name, raw_images, config, enforce_iframes):
        final_results = Results()
        all_required_regions = Results()
        low_phase_size = 0
        high_phase_size = 0
        nframes = sum(map(lambda e: "png" in e, os.listdir(raw_images)))

        self.init_server(nframes)

        for i in range(0, nframes, self.config.batch_size):
            start_frame = i
            end_frame = min(nframes, i + self.config.batch_size)
            self.logger.info(f"Processing frames {start_frame} to {end_frame}")

            # First iteration
            req_regions = Results()
            for fid in range(start_frame, end_frame):
                req_regions.append(Region(
                    fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
            batch_video_size, _ = compute_regions_size(
                req_regions, f"{vid_name}-base-phase", raw_images,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            low_phase_size += batch_video_size
            self.logger.info(f"{batch_video_size / 1024}KB sent in base phase."
                             f"Using QP {self.config.low_qp} and "
                             f"Resolution {self.config.low_resolution}.")
            results, rpn_regions = self.get_first_phase_results(vid_name)
            final_results.combine_results(
                results, self.config.intersection_threshold)
            all_required_regions.combine_results(
                rpn_regions, self.config.intersection_threshold)

            # Second Iteration
            if len(rpn_regions) > 0:
                batch_video_size, _ = compute_regions_size(
                    rpn_regions, vid_name, raw_images,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True)
                high_phase_size += batch_video_size
                self.logger.info(f"{batch_video_size / 1024}KB sent in second "
                                 f"phase. Using QP {self.config.high_qp} and "
                                 f"Resolution {self.config.high_resolution}.")
                results = self.get_second_phase_results(vid_name)
                final_results.combine_results(
                    results, self.config.intersection_threshold)

            # Cleanup for the next batch
            cleanup(vid_name, False, start_frame, end_frame)

        self.logger.info(f"Merging results")
        final_results = merge_boxes_in_results(
            final_results.regions_dict, 
            self.config.low_threshold, self.config.suppression_threshold)
        self.logger.info(f"Writing results for {vid_name}")
        final_results.fill_gaps(nframes)

        final_results.combine_results(
            all_required_regions, self.config.intersection_threshold)

        final_results.write(f"{vid_name}")

        return final_results, (low_phase_size, high_phase_size)
