import os
import re
import logging
import argparse
from backend.server import Server
from frontend.client import Client
from dds_utils import (ServerConfig, read_results_dict, Results, merge_boxes_in_results,
                       evaluate, write_stats)
from reduce_Bcost.streamB_infer import (analyze_dds_filtered, analyze_dds_filtered_resize, 
                        analyze_low_guide)
from reduce_Bcost.streamB_utils import (draw_region_rectangle)
from reduce_Bcost.resize_regions import (test_for_efficiency)
import ipdb
import shutil
import sys

from munch import *
import yaml

import coloredlogs

def main(args):
    coloredlogs.install(fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s", \
        level=args.verbosity.upper())

    logger = logging.getLogger("platform")
    logger.addHandler(logging.NullHandler())

    # Make simulation objects
    logger.info(f"Starting server with high threshold of "
                f"{args.high_threshold} low threshold of "
                f"{args.low_threshold} tracker length of "
                f"{args.tracker_length}")

    config = args
    server = None
    mode = None
    results, bw = None, None
    if args.req_regions_fname:
        mode = 'streamBpack'
        logger.warning("Reduce streamB cost")
        server = Server(config)
        req_regions = read_results_dict(args.req_regions_fname)
        # ipdb.set_trace()
        req_regions_result = Results()
        for fid, region_list in req_regions.items():
            for region_item in region_list:
                req_regions_result.append(region_item)

        if args.resize_method == 'resize_no':
            results, bw = analyze_dds_filtered(
                server, args.video_name, req_regions_result, args.low_results_path,  
                args.context_padding_type, args.context_val, args.blank_padding_type, args.blank_val,
                args.intersect_iou, args.merge_iou, args.filter_method,
                cleanup=args.cleanup, out_cost_file=args.out_cost_file
            )
        else:
            resize_max_area = float(args.resize_method.split('_')[1])
            results, bw = analyze_dds_filtered_resize(
                server, args.video_name, req_regions_result, args.low_results_path,  
                args.context_padding_type, args.context_val, args.blank_padding_type, args.blank_val,
                args.intersect_iou, args.merge_iou, resize_max_area, args.filter_method,
                cleanup=args.cleanup, out_cost_file=args.out_cost_file
            )
            # results, bw = analyze_low_guide(
            #     server, args.video_name, req_regions_result, args.low_results_path,  
            #     args.context_padding_type, args.context_val, args.blank_padding_type, args.blank_val,
            #     args.intersect_iou, args.merge_iou, resize_max_area, args.filter_method,
            #     cleanup=args.cleanup, out_cost_file=args.out_cost_file
            # )
        
    elif args.simulate:
        mode = "simulation"
        # ipdb.set_trace()
        logger.warning("Running DDS in SIMULATION mode")
        server = Server(config)

        logger.info("Starting client")
        client = Client(args.hname, config, server)
        # Run simulation
        logger.info(f"Analyzing video {args.video_name} with low resolution "
                    f"of {args.low_resolution} and high resolution of "
                    f"{args.high_resolution}")
        results, bw = client.analyze_video_simulate(
            args.video_name, args.low_images_path, args.high_images_path,
            args.high_results_path, args.low_results_path,
            args.enforce_iframes, args.mpeg_results_path,
            args.estimate_banwidth, args.debug_mode)
    elif not args.simulate and not args.hname and args.high_resolution != -1:
        mode = "emulation"
        logger.warning(f"Running DDS in EMULATION mode on {args.video_name}")
        server = Server(config)

        logger.info("Starting client")
        client = Client(args.hname, config, server)
        # Run emulation
        results, bw = client.analyze_video_emulate(
            args.video_name, args.high_images_path,
            args.enforce_iframes, args.low_results_path, args.debug_mode, 
            out_cost_file=args.out_cost_file)
    elif not args.simulate and not args.hname:
        mode = "mpeg"
        logger.warning(f"Running in MPEG mode with resolution "
                       f"{args.low_resolution} on {args.video_name}")
        server = Server(config)

        logger.info("Starting client")
        client = Client(args.hname, config, server)
        results, bw = client.analyze_video_merged_mpeg(
            args.video_name, args.high_images_path, args.enforce_iframes, args.single_frame_cnt,
            out_cost_file=args.out_cost_file)
    elif not args.simulate and args.hname:
        mode = "implementation"
        logger.warning(
            f"Running DDS using a server client implementation with "
            f"server running on {args.hname} using video {args.hname}")
        logger.info("Starting client")
        client = Client(args.hname, config, server)
        results, bw = client.analyze_video(
            args.video_name, args.high_images_path, config,
            args.enforce_iframes)

    # Evaluation and writing results
    # Read Groundtruth results
    low, high = bw
    f1 = 0
    stats = (0, 0, 0)
    number_of_frames = len(
        [x for x in os.listdir(args.high_images_path) if "png" in x])
    if args.ground_truth:
        ground_truth_dict = read_results_dict(args.ground_truth)
        logger.info("Reading ground truth results complete")
        tp, fp, fn, _, _, _, f1 = evaluate(
            number_of_frames - 1, results.regions_dict, ground_truth_dict,
            args.low_threshold, args.prune_score, args.max_object_size, args.max_object_size)
        stats = (tp, fp, fn)
        logger.info(f"Got an f1 score of {f1} "
                    f"for this experiment {mode} with "
                    f"tp {stats[0]} fp {stats[1]} fn {stats[2]} "
                    f"with total bandwidth {sum(bw)}")
    else:
        logger.info("No groundtruth given skipping evalution")

    # Write evaluation results to file
    write_stats(args.outfile, f"{args.video_name}", config, f1,
                stats, bw, number_of_frames, mode)


if __name__ == "__main__":

    # load configuration dictonary from command line
    # use munch to provide class-like accessment to python dictionary
    args = munchify(yaml.load(sys.argv[1], Loader=yaml.SafeLoader))

    if not args.simulate and not args.hname and args.high_resolution != -1:
        if not args.high_images_path:
            print("Running DDS in emulation mode requires raw/high "
                  "resolution images")
            exit()

    if not re.match("DEBUG|INFO|WARNING|CRITICAL", args.verbosity.upper()):
        print("Incorrect argument for verbosity."
              "Verbosity can only be one of the following:\n"
              "\tdebug\n\tinfo\n\twarning\n\terror")
        exit()

    if args.estimate_banwidth and not args.high_images_path:
        print("DDS needs location of high resolution images to "
              "calculate true bandwidth estimate")
        exit()

    if args.high_resolution == -1:
        print("Only one resolution given, running MPEG emulation")
        assert args.high_qp == -1, "MPEG emulation only support one QP"
    else:
        assert args.low_resolution <= args.high_resolution, \
                f"The resolution of low quality({args.low_resolution})"\
                f"can't be larger than high quality({args.high_resolution})"
        assert not(args.low_resolution == args.high_resolution and 
                    args.low_qp < args.high_qp),\
                f"Under the same resolution, the QP of low quality({args.low_qp})"\
                f"can't be higher than the QP of high quality({args.high_qp})"

    main(args)
