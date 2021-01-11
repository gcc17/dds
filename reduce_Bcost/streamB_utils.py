import numpy as np
import os
import cv2 as cv
import ipdb
import shutil
import networkx
from networkx.algorithms.components.connected import connected_components
from dds_utils import (Region, compute_regions_size, Results, evaluate, read_results_dict)
from .new_region import (PadShiftRegion)
import re
import subprocess


def normalize_image(img_arr):
    img_channel_mean = [0.485, 0.456, 0.406]
    # [123.68, 116.779, 103.939] RGB-order
    # 255*0.485 = 123.675
    for channel_idx in range(3):
        img_arr[:,:,channel_idx] = np.uint8(255 * img_channel_mean[channel_idx])
    return img_arr


def find_region_same_id(region_id, pad_regions_list):
    # find corresponding ShiftRegion in pad_regions_list
    for idx, single_shift_region in enumerate(pad_regions_list):
        if single_shift_region.region_id == region_id:
            return idx
    return None


def get_percentile(regions_list, req_percent):
    area_list = []
    for cur_region in regions_list:
        cur_region_area = cur_region.w * cur_region.h
        area_list.append(cur_region_area)
    area_np = np.array(area_list)
    if 0 <= req_percent <= 1:
        req_percent *= 100
    if req_percent < 0 or req_percent > 100:
        print('Error percent')
        return None
    return np.percentile(area_np, req_percent)


def greater_than_target(target_val, cur_val, choose_metric='max'):
    if choose_metric == 'max':
        return (cur_val >= target_val)
    if choose_metric == 'min':
        return (cur_val <= target_val)


def draw_region_rectangle(image_direc, fnames, regions_dict, save_image_direc,
        rec_color=(255, 0, 0), rec_side_width=2, anno_text=None, display_result=False,
        text_loc=(50,500), font_scale=0.75, thickness=1, drop_no_rect=False, clean_save=True):
    os.makedirs(save_image_direc, exist_ok=True)
    if clean_save:
        for fname in os.listdir(save_image_direc):
            if "png" in fname:
                os.remove(os.path.join(save_image_direc, fname))
            
    for fname in fnames:
        if "png" not in fname:
            continue
        fid = int(fname.split(".")[0])
        if drop_no_rect and ( (fid not in regions_dict.keys()) or (not regions_dict[fid])) :
            continue

        image_path = os.path.join(image_direc, fname)
        if not os.path.exists(image_path):
            continue
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if fid in regions_dict.keys():
            for region in regions_dict[fid]:
                if region.w * region.h == 0:
                    continue
                image_h = image.shape[0]
                image_w = image.shape[1]
                x_min = int(region.x * image_w)
                y_min = int(region.y * image_h)
                x_max = int(region.w * image_w) + x_min
                y_max = int(region.h * image_h) + y_min
                # ipdb.set_trace()
                cv.rectangle(image, (x_min, y_min), (x_max, y_max), rec_color, rec_side_width)

                if display_result:
                    anno_font = cv.FONT_HERSHEY_SIMPLEX
                    result_text = f"{region.label}-{round(region.conf, 2)}"
                    cv.putText(image, result_text, (x_min, y_min), anno_font, font_scale, rec_color, thickness)
        
        if anno_text:
            anno_font = cv.FONT_HERSHEY_SIMPLEX
            new_text = f"{anno_text}-{fid}"
            cv.putText(image, new_text, text_loc, anno_font, font_scale, rec_color, thickness)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(save_image_direc, fname), image, 
            [cv.IMWRITE_PNG_COMPRESSION, 0])


def filter_true_regions(region_list, confid_thresh, max_area_thresh, relevant_classes=["vehicle"]):
    true_region_list = []
    for r in region_list:
        if r.conf > confid_thresh and r.w*r.h <= max_area_thresh and r.label in relevant_classes:
            true_region_list.append(r)
    return true_region_list
        

def region_iou(region1, region2):
    x3 = max(region1.x, region2.x)
    y3 = max(region1.y, region2.y)
    x4 = min(region1.x + region1.w, region2.x + region2.w)
    y4 = min(region1.y + region1.h, region2.y + region2.h)
    
    if x3 > x4 or y3 > y4:
        return 0
    else:
        overlap_area = (x4-x3)*(y4-y3)
        total_area = region1.w * region1.h + region2.w * region2.h - overlap_area
        return overlap_area / total_area


def two_results_diff(max_fid, regions_dict1, regions_dict2, confid_thresh1, confid_thresh2, 
        max_area_thresh1, max_area_thresh2, iou_thresh=0.3, vis_common=0, drop_no_rect=True,
        src_image_direc=None, diff_image_direc=None, name1=None, name2=None):
    t1f2 = {}
    t2f1 = {}
    t1t2 = {}
    t2t1 = {}
    for fid in range(max_fid+1):
        t1f2[fid] = []
        t2f1[fid] = []
        t1t2[fid] = []
        t2t1[fid] = []

        if fid not in regions_dict1.keys():
            if fid not in regions_dict2.keys():
                continue
            bbox2 = filter_true_regions(regions_dict2[fid], confid_thresh2, max_area_thresh2)
            t2f1[fid] = bbox2
            continue
        elif fid not in regions_dict2.keys():
            bbox1 = filter_true_regions(regions_dict1[fid], confid_thresh1, max_area_thresh1)
            t1f2[fid] = bbox1
            continue
        
        bbox1 = regions_dict1[fid]
        bbox2 = regions_dict2[fid]
        bbox1 = filter_true_regions(bbox1, confid_thresh1, max_area_thresh1)
        bbox2 = filter_true_regions(bbox2, confid_thresh2, max_area_thresh2)
        
        for region1 in bbox1:
            found = False
            for region2 in bbox2:
                if region_iou(region1, region2) > iou_thresh:
                    found = True
                    break
            if not found:
                t1f2[fid].append(region1)
            else:
                t1t2[fid].append(region1)
        for region2 in bbox2:
            found = False
            for region1 in bbox1:
                if region_iou(region1, region2) > iou_thresh:
                    found = True
                    break
            if not found:
                t2f1[fid].append(region2)
            else:
                t2t1[fid].append(region2)

    if src_image_direc and diff_image_direc:
        # find src images names
        fnames = []
        for fid in range(max_fid+1):
            fnames.append(f"{str(fid).zfill(10)}.png")
        diff_tmp_direc = os.path.join(src_image_direc, "diff_tmp")
        if name1 and name2:
            text1 = f"true-{name1}-false-{name2}"
            text2 = f"true-{name2}-false-{name1}"
            text3 = f"both_true-{name1}"
            text4 = f"both_true-{name2}"
        else:
            text1 = "true1-false2"
            text2 = "true2-false1"
            text3 = "both_true1"
            text4 = "both_true2"
        draw_region_rectangle(src_image_direc, fnames, t1f2, 
            diff_tmp_direc, anno_text=text1)
        if vis_common == 0:
            draw_region_rectangle(diff_tmp_direc, fnames, t2f1, 
                diff_image_direc, rec_color=(0,0,255), anno_text=text2, text_loc=(50,550), 
                drop_no_rect=drop_no_rect)
        else:
            diff_tmp_direc2 = os.path.join(src_image_direc, "diff_tmp2")
            draw_region_rectangle(diff_tmp_direc, fnames, t2f1, 
                diff_tmp_direc2, rec_color=(0,0,255), anno_text=text2, text_loc=(50,550))
            if vis_common == 1:
                draw_region_rectangle(diff_tmp_direc2, fnames, t1t2, 
                    diff_image_direc, rec_color=(0,255,0), anno_text=text3, text_loc=(50,450))
            elif vis_common == 2:
                draw_region_rectangle(diff_tmp_direc2, fnames, t2t1, 
                    diff_image_direc, rec_color=(0,255,0), anno_text=text4, text_loc=(50,450))
            else:
                diff_tmp_direc3 = os.path.join(src_image_direc, "diff_tmp3")
                draw_region_rectangle(diff_tmp_direc2, fnames, t2t1, 
                    diff_tmp_direc3, rec_color=(0,255,0), anno_text=text4, text_loc=(50,450))
                draw_region_rectangle(diff_tmp_direc3, fnames, t1t2, 
                    diff_image_direc, rec_color=(0,255,255), anno_text=text3, text_loc=(50,400))
                shutil.rmtree(diff_tmp_direc3)

            shutil.rmtree(diff_tmp_direc2)

        shutil.rmtree(diff_tmp_direc)
    
    return t1f2, t2f1, t1t2, t2t1


def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G


def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current


def filter_bbox_group(bb1, bb2, iou_threshold):
    if region_iou(bb1, bb2) > iou_threshold:
        return True
    return False


def pairwise_overlap_indexing_list(single_result_frame, iou_threshold):
    pointwise = [[i] for i in range(len(single_result_frame))]
    pairwise = [[i, j] for i, x in enumerate(single_result_frame)
                for j, y in enumerate(single_result_frame)
                if i != j if filter_bbox_group(x, y, iou_threshold)]
    return pointwise + pairwise


def merge_small_regions(single_result_frame, merged_region_id, index_to_merge):
    # directly using the largest box
    merged_new_regions_list = []
    merged_region_contain_dict = {}
    cur_merged_region_id = merged_region_id
    is_new = False
        
    for i in index_to_merge:
        i2np = np.array([j for j in i])
        left = min(np.array(single_result_frame)[i2np], key=lambda x: x.x)
        top = min(np.array(single_result_frame)[i2np], key=lambda x: x.y)
        right = max(
            np.array(single_result_frame)[i2np], key=lambda x: x.x + x.w)
        bottom = max(
            np.array(single_result_frame)[i2np], key=lambda x: x.y + x.h)

        # merge small PadShiftRegion into a large Region
        if isinstance(left, PadShiftRegion):
            fid = left.original_region.fid
            is_new = True
        else:
            fid = left.fid
        fid, x, y, w, h, conf, label, resolution, origin = (
            fid, left.x, top.y, right.x + right.w - left.x,
            bottom.y + bottom.h - top.y, 1, 'object', 1.0, 'merged')
        single_merged_region = Region(fid, x, y, w, h, conf,
                                      label, resolution, origin)
        
        if is_new:
            # transform this merged region into a PadShiftRegion object
            merged_new_region = PadShiftRegion(single_merged_region, cur_merged_region_id, x, y, w, h)
            merged_new_regions_list.append(merged_new_region)
            # store small region_ids in merged large region
            contain_small_region_ids = []
            for j in i:
                small_region_id = single_result_frame[j].region_id
                contain_small_region_ids.append(small_region_id)
            merged_region_contain_dict[cur_merged_region_id] = contain_small_region_ids
            cur_merged_region_id += 1
        else:
            merged_new_regions_list.append(single_merged_region)
    
    return merged_new_regions_list, merged_region_contain_dict


def merge_one_frame_regions(new_regions_list, merged_region_id, iou_threshold):
    # merge boxes IoU larger than threshold
    # find region pair with large IoU
    overlap_pairwise_list = pairwise_overlap_indexing_list(new_regions_list, iou_threshold)
    # transform into graph: each small region is a 
    overlap_graph = to_graph(overlap_pairwise_list)
    # find connected component of the graph
    grouped_bbox_idx = [c for c in sorted(
        connected_components(overlap_graph), key=len, reverse=True
    )]
    merged_new_regions_list, merged_region_contain_dict = \
        merge_small_regions(new_regions_list, merged_region_id, grouped_bbox_idx)
    
    return merged_new_regions_list, merged_region_contain_dict


def write_compute_cost_txt(fname, vid_name, frame_cnt, 
        pad_time, shift_time, infer_time, total_time):
    header = ("video-name,frame-count,pad-time,shift-time,infer-time,total-time")
    cost = (f"{vid_name},{frame_cnt},{pad_time},{shift_time},{infer_time},{total_time}")

    if not os.path.isfile(fname):
        str_to_write = f"{header}\n{cost}\n"
    else:
        str_to_write = f"{cost}\n"

    with open(fname, "a") as f:
        f.write(str_to_write)


def write_compute_cost_csv(fname, vid_name, frame_cnt, 
        pad_time, shift_time, infer_time, total_time):
    header = ("video-name,frame-count,pad-time,shift-time,infer-time,total-time").split(",")
    cost = (f"{vid_name},{frame_cnt},{pad_time},{shift_time},{infer_time},{total_time}").split(",")

    results_files = open(fname, "a")
    csv_writer = csv.writer(results_files)
    if not os.path.isfile(fname):
        # If file does not exist write the header row
        csv_writer.writerow(header)
    csv_writer.writerow(cost)
    results_files.close()


def write_compute_cost(fname, vid_name, frame_cnt, pad_time, shift_time, infer_time, total_time):
    if re.match(r"\w+[.]csv\Z", fname):
        write_compute_cost_csv(fname, vid_name, frame_cnt, 
            pad_time, shift_time, infer_time, total_time)
    else:
        write_compute_cost_txt(fname, vid_name, frame_cnt, 
            pad_time, shift_time, infer_time, total_time)


def extract_save_image_from_video(video_direc, save_image_direc):
    if not os.path.isdir(video_direc):
        return
    os.makedirs(save_image_direc, exist_ok=True)

    for fname in os.listdir(save_image_direc):
        if "png" not in fname:
            continue
        else:
            os.remove(os.path.join(save_image_direc, fname))
    encoded_vid_path = os.path.join(video_direc, "temp.mp4")
    extracted_images_path = os.path.join(save_image_direc, "%010d.png")
    decoding_result = subprocess.run(["ffmpeg", "-y",
                                      "-i", encoded_vid_path,
                                      "-pix_fmt", "yuvj420p",
                                      "-g", "8", "-q:v", "2",
                                      "-vsync", "0", "-start_number", "0",
                                      extracted_images_path],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
    if decoding_result.returncode != 0:
        print("DECODING FAILED")
        print(decoding_result.stdout)
        print(decoding_result.stderr)
        exit()


def prepare_high_low_images(src_image_direc, high_image_direc, low_image_direc,
        high_resolution, high_qp, low_resolution, low_qp, enforce_iframes):
    base_req_regions = Results()
    fnames = sorted([f for f in os.listdir(src_image_direc) if "png" in f])
    for fname in fnames:
        fid = int(fname.split(".")[0])
        base_req_regions.append(Region(fid, 0, 0, 1, 1, 1.0, 2, high_resolution))

    encoded_high_all_video_size, high_pixel_size = compute_regions_size(
            base_req_regions, high_image_direc, src_image_direc, 
            high_resolution, high_qp,
            enforce_iframes, True)
    encoded_low_all_video_size, low_pixel_size = compute_regions_size(
        base_req_regions, low_image_direc, src_image_direc,
        low_resolution, low_qp, 
        enforce_iframes, True)
    high_video_direc = f"{high_image_direc}-cropped"
    extract_save_image_from_video(high_video_direc, high_image_direc)
    low_video_direc = f"{low_image_direc}-cropped"
    extract_save_image_from_video(low_video_direc, low_image_direc)

    shutil.rmtree(high_video_direc)
    shutil.rmtree(low_video_direc)


def get_clean_result(result_dict, confid_thresh, max_area_thresh):
    clean_dict = {}
    for fid, result_list in result_dict.items():
        clean_result_list = filter_true_regions(result_list, confid_thresh, max_area_thresh)
        clean_dict[fid] = clean_result_list
    return clean_dict


def get_new_gt(total_result_dict, low_result_dict, iou_thresh, gt_confid_thresh=0.3, gt_max_area_thresh=0.3, \
        low_confid_thresh=0.5, low_max_area_thresh=0.3):
    new_result_dict = {}
    for fid, frame_total_list in total_result_dict.items():
        new_result_dict[fid] = []
        frame_total_list = filter_true_regions(frame_total_list, gt_confid_thresh, gt_max_area_thresh)
        frame_low_list = filter_true_regions(low_result_dict[fid], low_confid_thresh, low_max_area_thresh)
        for total_region in frame_total_list:
            find_overlap = False
            for low_region in frame_low_list:
                if region_iou(total_region, low_region) >= iou_thresh:
                    find_overlap = True
                    break
            if not find_overlap:
                new_result_dict[fid].append(total_region)
                    

    return new_result_dict


def new_evaluate(new_gt_dict, high_pack_result_dict, gt_confid_thresh, pack_confid_thresh,
        gt_max_area_thresh, pack_max_area_thresh, iou_thresh=0.3):
    max_fid = max( list(new_gt_dict.keys()) )
    max_fid = int(max_fid)
    return evaluate(
        max_fid, high_pack_result_dict, new_gt_dict, gt_confid_thresh, pack_confid_thresh, 
        gt_max_area_thresh, pack_max_area_thresh, iou_thresh
    )


def enumerate_files(file_direc):
    fnames = []
    for fname in os.listdir(file_direc):
        if 'pack-txt' in fname:
            fnames.append(fname)
        if 'high_phase_results' in fname:
            fnames.append(fname)
        if 'mpeg' in fname:
            fnames.append(fname)

    return fnames

def get_gt_file(file_direc, dds_as_gt):
    if dds_as_gt:
        for fname in os.listdir(file_direc):
            if 'high_phase_results' in fname and 'dds' in fname:
                return fname
    else:
        for fname in os.listdir(file_direc):
            if 'gt' in fname:
                return fname

def get_low_file(file_direc):
    for fname in os.listdir(file_direc):
        if 'mpeg' in fname and '36' in fname:
            return fname

def get_high_file(file_direc):
    for fname in os.listdir(file_direc):
        if 'mpeg' in fname and '26' in fname:
            return fname

def get_dds_file(file_direc):
    for fname in os.listdir(file_direc):
        if 'dds' in fname and \
            not ('high_phase_results' in fname or 'req_regions' in fname):
            return fname

def fname_2_methodname(fname):
    if fname.split('_')[2] == 'mpeg':
        return '_'.join(fname.split('_')[2:])
    if fname.split('-')[1] == 'high_phase_results':
        return 'dds'
    fname = fname.split('-')[0]
    para_list = fname.split('_')
    method_name = '_'.join(para_list[3:])
    return method_name

def choose_target_metric(tp, fp, fn, f1, stats_metric):
    if stats_metric == 'F1':
        return f1
    elif stats_metric == 'TP':
        return tp
    elif stats_metric == 'FP':
        return fp
    elif stats_metric == 'FN':
        return fn

def get_req_file(file_direc):
    for fname in os.listdir(file_direc):
        if 'req_regions' in fname:
            return fname
        

def new_evaluate_all(file_direc, video_name=None, dds_as_gt=False, stats_metric='F1', \
        gt_confid_thresh=0.3, pack_confid_thresh=0.5, gt_max_area_thresh=0.3, pack_max_area_thresh=0.3, 
        iou_thresh=0.3, new_stats_dict={}, \
        vis_gt=False, src_image_direc=None, save_image_direc=None):
    fnames = enumerate_files(file_direc)
    gt_fname = get_gt_file(file_direc, dds_as_gt)
    total_gt_dict = read_results_dict(os.path.join(file_direc, gt_fname))
    print(f'gt_fname: {gt_fname}')
    low_fname = get_low_file(file_direc)
    low_result_dict = read_results_dict(os.path.join(file_direc, low_fname))
    print(f'low_fname: {low_fname}')
    if dds_as_gt:
        new_gt_dict = get_new_gt(total_gt_dict, low_result_dict, iou_thresh, \
            pack_confid_thresh, pack_max_area_thresh, pack_confid_thresh, pack_max_area_thresh)
    else:
        new_gt_dict = get_new_gt(total_gt_dict, low_result_dict, iou_thresh, \
            gt_confid_thresh, gt_max_area_thresh, pack_confid_thresh, pack_max_area_thresh)
    
    if vis_gt and src_image_direc and save_image_direc:
        img_fnames = sorted([f for f in os.listdir(src_image_direc) if "png" in f])
        tmp_direc = f'{save_image_direc}-tmp'
        os.makedirs(tmp_direc, exist_ok=True)
        draw_region_rectangle(src_image_direc, img_fnames, new_gt_dict, tmp_direc, rec_side_width=4)

        tmp1 = f'{save_image_direc}-tmp1'
        tmp2 = f'{save_image_direc}-tmp2'
        os.makedirs(tmp1, exist_ok=True)
        os.makedirs(tmp2, exist_ok=True)
        clean_total_gt = get_clean_result(total_gt_dict, gt_confid_thresh, gt_max_area_thresh)
        clean_low_gt = get_clean_result(low_result_dict, pack_confid_thresh, pack_max_area_thresh)
        draw_region_rectangle(tmp_direc, img_fnames, clean_total_gt, tmp1, rec_color=(0,255,0), \
            anno_text='total_gt')
        draw_region_rectangle(tmp1, img_fnames, clean_low_gt, tmp2, rec_color=(0,255,255), \
            anno_text='low_gt', text_loc=(50,450))
        shutil.rmtree(tmp_direc)
        shutil.rmtree(tmp1)

        req_fname = get_req_file(file_direc)
        req_regions_dict = read_results_dict(os.path.join(file_direc, req_fname))
        os.makedirs(save_image_direc, exist_ok=True)
        draw_region_rectangle(tmp2, img_fnames, req_regions_dict, save_image_direc, rec_color=(0,0,255))
        shutil.rmtree(tmp2)

        # import ipdb; ipdb.set_trace()

    if not video_name:
        video_name = os.path.split(file_direc)[1]
    max_fid = max(list(total_gt_dict.keys()))

    for fname in fnames:
        high_pack_result_dict = read_results_dict(os.path.join(file_direc, fname))
        method_name = fname_2_methodname(fname)
        if method_name == 'dds' or 'mpeg' in method_name:
            high_pack_result_dict = get_new_gt(high_pack_result_dict, low_result_dict, iou_thresh, \
                pack_confid_thresh, pack_max_area_thresh, pack_confid_thresh, pack_max_area_thresh)

        tp, fp, fn, _, _, _, f1 = new_evaluate(
            new_gt_dict, high_pack_result_dict, gt_confid_thresh, pack_confid_thresh,
            gt_max_area_thresh, pack_max_area_thresh, iou_thresh
        )
        target_metric = choose_target_metric(tp, fp, fn, f1, stats_metric)

        if method_name not in new_stats_dict.keys():
            new_stats_dict[method_name] = {}
        new_stats_dict[method_name][video_name] = target_metric

        out_stats = os.path.join(file_direc, 'new_stats')
        if os.path.exists(out_stats):
            os.remove(out_stats)
        cur_fname = fname.split('-')[0]
        cur_video_name = f'{os.path.join("results", video_name, cur_fname)}'
        write_new_stats(out_stats, cur_video_name, tp, fp, fn, f1)


def write_new_stats(out_stats, cur_video_name, tp, fp, fn, f1):
    if re.match(r"\w+[.]csv\Z", out_stats):
        write_stats_csv(out_stats, cur_video_name, tp, fp, fn, f1)
    else:
        write_stats_txt(out_stats, cur_video_name, tp, fp, fn, f1)

def write_stats_csv(out_stats, cur_video_name, tp, fp, fn, f1):
    header = ("video-name,TP,FP,FN,F1").split(",")
    stats = (f"{cur_video_name},{tp},{fp},{fn},{f1}").split(",")
    results_files = open(out_stats, "a")
    csv_writer = csv.writer(results_files)
    if not os.path.isfile(out_stats):
        # If file does not exist write the header row
        csv_writer.writerow(header)
    csv_writer.writerow(stats)
    results_files.close()

def write_stats_txt(out_stats, cur_video_name, tp, fp, fn, f1):
    header = ("video-name,TP,FP,FN,F1")
    stats = (f"{cur_video_name},{tp},{fp},{fn},{f1}")

    if not os.path.isfile(out_stats):
        str_to_write = f"{header}\n{stats}\n"
    else:
        str_to_write = f"{stats}\n"

    with open(out_stats, "a") as f:
        f.write(str_to_write)


def filter_bad_parameters(stats_dict, costs_dict, stats_metric, compare_mpeg=True):
    dds_stats_video_dict = stats_dict['dds']
    mpeg_stats_video_dict = stats_dict['mpeg_1.0_26']
    dds_costs_video_dict = costs_dict['dds']
    mpeg_costs_video_dict = costs_dict['mpeg_1.0_26']
    if stats_metric in ['F1', 'TP']:
        choose_metric = 'max'
    elif stats_metric in ['FP', 'FN']:
        choose_metric = 'min'

    to_remove_methods = []
    # remove methods: cost larger than mpeg_26 and stat "smaller" than mpeg_26
    for method_name, method_stats_video_dict in stats_dict.items():
        if method_name in ['dds', 'mpeg_1.0_26']:
            continue
        method_costs_video_dict = costs_dict[method_name]
        to_remove = True
        for cur_video_name in method_stats_video_dict.keys():
            if compare_mpeg:
                cur_mpeg_stat = mpeg_stats_video_dict[cur_video_name]
                cur_mpeg_cost = mpeg_costs_video_dict[cur_video_name]

                cur_stat = method_stats_video_dict[cur_video_name]
                cur_cost = method_costs_video_dict[cur_video_name]
                if cur_cost < cur_mpeg_cost and greater_than_target(cur_mpeg_stat, cur_stat, choose_metric):
                    to_remove = False
                    break
            else:
                cur_dds_stat = dds_stats_video_dict[cur_video_name]
                cur_dds_cost = dds_costs_video_dict[cur_video_name]

                cur_stat = method_stats_video_dict[cur_video_name]
                cur_cost = method_costs_video_dict[cur_video_name]
                if cur_cost < cur_dds_cost and greater_than_target(cur_dds_stat, cur_stat, choose_metric):
                    to_remove = False
                    break

        if to_remove:
            to_remove_methods.append(method_name)
        
    return to_remove_methods
    

def filter_bad_parameters_wrapper(server_result_direc, all_target_videos, costs_metric, stats_metric):
    from .plot_utils import (read_logs, pick_list_item)
    costs_dict = {}
    new_stats_dict = {}
    for target_video_name in all_target_videos:
        file_direc = os.path.join(server_result_direc, target_video_name)
        costs_list = read_logs(os.path.join(file_direc, 'costs'))
        costs_result_dict, costs_baseline_result_dict = pick_list_item(costs_list, costs_metric)

        for cost_key in costs_result_dict.keys():
            if cost_key not in costs_dict.keys():
                costs_dict[cost_key] = {}
            costs_dict[cost_key].update(costs_result_dict[cost_key])
        for cost_key in costs_baseline_result_dict.keys():
            if cost_key not in costs_dict.keys():
                costs_dict[cost_key] = {}
            costs_dict[cost_key].update(costs_baseline_result_dict[cost_key])
        
        new_evaluate_all(file_direc, target_video_name, new_stats_dict=new_stats_dict, stats_metric=stats_metric)
    
    print(new_stats_dict)
    to_remove_methods = filter_bad_parameters(new_stats_dict, costs_dict, stats_metric, compare_mpeg=False)
    # print(to_remove_methods)
    all_methods = list(new_stats_dict.keys())
    left_methods = []
    for cur_method in all_methods:
        if cur_method not in to_remove_methods:
            left_methods.append(cur_method)
    print(left_methods)


def merge_images_by_frame(req_regions_dict, high_image_direc, low_image_direc, save_image_direc):
    os.makedirs(save_image_direc, exist_ok=True)
    for fid in req_regions_dict.keys():
        fname = f'{str(fid).zfill(10)}.png'
        # Read high resolution image
        high_image = cv.imread(os.path.join(high_image_direc, fname))
        width = high_image.shape[1]
        height = high_image.shape[0]
        # Read low resolution image
        low_image = cv.imread(os.path.join(low_image_direc, fname))
        # Enlarge low resolution image
        enlarged_image = cv.resize(low_image, (width, height), fx=0, fy=0,
                                   interpolation=cv.INTER_CUBIC)
        # Put regions in place
        for r in req_regions_dict[fid]:
            if fid != r.fid:
                continue
            x0 = int(r.x * width)
            y0 = int(r.y * height)
            x1 = int((r.w * width) + x0 - 1)
            y1 = int((r.h * height) + y0 - 1)

            enlarged_image[y0:y1, x0:x1, :] = high_image[y0:y1, x0:x1, :]
        cv.imwrite(os.path.join(save_image_direc, fname), enlarged_image,
                   [cv.IMWRITE_PNG_COMPRESSION, 0])


def track2req_region(track_new_regions_dict, req_regions_dict):
    fid_list = []
    for track_new_region in track_new_regions_dict.values():
        if track_new_region.original_region.fid not in fid_list:
            fid_list.append(track_new_region.original_region.fid)
    
    new_req_regions_dict = {}
    for req_fid in fid_list:
        new_req_regions_dict[req_fid] = req_regions_dict[req_fid]
    
    return new_req_regions_dict, fid_list


def exclude_frame_regions(merged_new_regions_dict, fid_list):
    exclude_new_regions_dict = {}
    for merged_region_id, merged_new_region in merged_new_regions_dict.items():
        if merged_new_region.original_region.fid in fid_list:
            continue
        exclude_new_regions_dict[merged_region_id] = merged_new_region
    
    return exclude_new_regions_dict

