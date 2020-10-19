"""
    entrance.py - user entrance for the platform
    author: Qizheng Zhang (qizhengz@uchicago.edu)
            Kuntai Du (kuntai@uchicago.edu)
"""

import os
import subprocess
import yaml
import sys

# dirty fix
sys.path.append('../')

def load_configuration():
    """read configuration information from yaml file

    Returns:
        dict: information of the yaml file
    """
    with open('configuration.yml', 'r') as config:
        config_info = yaml.load(config, Loader=yaml.FullLoader)
    return config_info


def execute_single(single_instance):
    """execute an atomic instance

    Args:
        single_instance (dict): the instance to be executed
    """
    # unpacking
    baseline = single_instance['method']

    # branching based on baselines
    if baseline == 'gt':
        # unpacking
        video_name = single_instance['video_name']
        original_images_dir = os.path.join(data_dir, video_name, 'src')
        result_direc = os.path.join("results", video_name)

        # skip if result file already exists
        result_file_name = f"{video_name}_gt"
        if single_instance['overwrite'] == False and \
            os.path.exists(os.path.join(result_direc, result_file_name)):
            print(f"Skipping {result_file_name}")
        # otherwise, start execution
        else:
            single_instance['video_name'] = os.path.join(result_direc, result_file_name)
            single_instance['high_images_path'] = f'{original_images_dir}'
            single_instance['outfile'] = os.path.join(result_direc, 'stats')
            single_instance['out_cost_file'] = os.path.join(result_direc, 'costs')

            subprocess.run(['python', '../play_video.py', 
                            yaml.dump(single_instance)])

    # assume we are running emulation
    elif baseline == 'mpeg':
        # unpacking
        video_name = single_instance['video_name']
        mpeg_qp = single_instance['low_qp']
        mpeg_resolution = single_instance['low_resolution']
        original_images_dir = os.path.join(data_dir, video_name, 'src')
        result_direc = os.path.join("results", video_name)

        # skip if result file already exists
        result_file_name = f"{video_name}_mpeg_{mpeg_resolution}_{mpeg_qp}"
        if single_instance['overwrite'] == False and \
            os.path.exists(os.path.join(result_direc, result_file_name)):
            print(f"Skipping {result_file_name}")
        else:
            single_instance['video_name'] = os.path.join(result_direc, result_file_name)
            single_instance['high_images_path'] = f'{original_images_dir}'
            single_instance['outfile'] = os.path.join(result_direc, 'stats')
            single_instance['out_cost_file'] = os.path.join(result_direc, 'costs')
            single_instance['ground_truth'] = os.path.join(result_direc, f'{video_name}_gt')

            subprocess.run(['python', '../play_video.py',
                            yaml.dump(single_instance)])

    elif baseline == 'dds':
        # unpacking
        video_name = single_instance['video_name']
        original_images_dir = os.path.join(data_dir, video_name, 'src')
        result_direc = os.path.join("results", video_name)
        low_qp = single_instance['low_qp']
        high_qp = single_instance['high_qp']
        low_res = single_instance['low_resolution']
        high_res = single_instance['high_resolution']
        rpn_enlarge_ratio = single_instance['rpn_enlarge_ratio']
        batch_size = single_instance['batch_size']
        prune_score = single_instance['prune_score']
        objfilter_iou = single_instance['objfilter_iou']
        size_obj = single_instance['size_obj']

        # skip if result file already exists
        # You could customize the way to serialize the parameters into filename by yourself
        result_file_name = (f"{video_name}_dds_{low_res}_{high_res}_{low_qp}_{high_qp}_"
                            f"{rpn_enlarge_ratio}_twosides_batch_{batch_size}_"
                            f"{prune_score}_{objfilter_iou}_{size_obj}")
        if single_instance['overwrite'] == False and \
            os.path.exists(os.path.join(result_direc, result_file_name)):
            print(f"Skipping {result_file_name}")
        else:
            single_instance['video_name'] = os.path.join(result_direc, result_file_name)
            single_instance['high_images_path'] = f'{original_images_dir}'
            single_instance['outfile'] = os.path.join(result_direc, 'stats')
            single_instance['out_cost_file'] = os.path.join(result_direc, 'costs')
            single_instance['ground_truth'] = os.path.join(result_direc, f'{video_name}_gt')
            single_instance['low_results_path'] = os.path.join(result_direc, f'{video_name}_mpeg_{low_res}_{low_qp}')

            if single_instance["mode"] == 'implementation':
                assert single_instance['hname'] != False, "Must provide the server address for implementation, abort."
                # single_instance['hname'] = '127.0.0.1:5001'
                
            subprocess.run(['python', '../play_video.py',
                                yaml.dump(single_instance)])
    
    elif baseline == 'streamBpack':
        # unpacking
        video_name = single_instance['video_name']
        original_images_dir = os.path.join(data_dir, video_name, 'src')
        result_direc = os.path.join("results", video_name)
        low_qp = single_instance['low_qp']
        high_qp = single_instance['high_qp']
        low_res = single_instance['low_resolution']
        high_res = single_instance['high_resolution']
        rpn_enlarge_ratio = single_instance['rpn_enlarge_ratio']
        batch_size = single_instance['batch_size']
        prune_score = single_instance['prune_score']
        objfilter_iou = single_instance['objfilter_iou']
        size_obj = single_instance['size_obj']
        dds_result_file_name = (f"{video_name}_dds_{low_res}_{high_res}_{low_qp}_{high_qp}_"
                            f"{rpn_enlarge_ratio}_twosides_batch_{batch_size}_"
                            f"{prune_score}_{objfilter_iou}_{size_obj}")

        context_padding_type = single_instance['context_padding_type']
        context_val = single_instance['context_val']
        blank_padding_type = single_instance['blank_padding_type']
        blank_val = single_instance['blank_val']
        intersect_iou = single_instance['intersect_iou']
        merge_iou = single_instance['merge_iou']
        track_region = single_instance['track_region']
        if not track_region:
            max_diff = 0.0
            max_frame_interval = 0
        else:
            max_diff = single_instance['max_diff']
            max_frame_interval = single_instance['max_frame_interval']
        percent_list = single_instance['percent_dict']['percent_list']
        qp_list = single_instance['qp_dict']['qp_list']
        if len(qp_list) != len(percent_list):
            print('qp and percent list do not match')
            return
        single_instance['percent_list'] = percent_list
        single_instance['qp_list'] = qp_list

        # skip if result file already exists
        # You could customize the way to serialize the parameters into filename by yourself
        result_file_name = (f'{video_name}_streamBpack_{context_padding_type}_{context_val}_'
                            f'{blank_padding_type}_{blank_val}_inter_{intersect_iou}_merge_{merge_iou}_'
                            f'track_{track_region}_diff_{max_diff}_frameinter_{max_frame_interval}_'
                            f'{qp_list}_{percent_list}')
        if single_instance['overwrite'] == False and \
            os.path.exists(os.path.join(result_direc, f'{result_file_name}-pack-txt')):
            print(f"Skipping {result_file_name}")
        else:
            single_instance['video_name'] = os.path.join(result_direc, result_file_name)
            single_instance['high_images_path'] = f'{original_images_dir}'
            single_instance['outfile'] = os.path.join(result_direc, 'stats')
            single_instance['out_cost_file'] = os.path.join(result_direc, 'costs')
            single_instance['ground_truth'] = os.path.join(result_direc, f'{video_name}_gt')
            single_instance['low_results_path'] = os.path.join(result_direc, f'{video_name}_mpeg_{low_res}_{low_qp}')
            single_instance['req_regions_fname'] = os.path.join(result_direc, \
                f'{dds_result_file_name}-req_regions')

            if single_instance["mode"] == 'implementation':
                assert single_instance['hname'] != False, "Must provide the server address for implementation, abort."
                # single_instance['hname'] = '127.0.0.1:5001'
                
            subprocess.run(['python', '../play_video.py',
                                yaml.dump(single_instance)])


def parameter_sweeping(instances, new_instance, keys):
    """recursive function for parameter sweeping

    Args:
        instances (dict): the instance in process
        new_instance (dict): recursive parameter
        keys (list): keys of the instance in process
    """
    if keys == []: # base case
        execute_single(new_instance)
    else: # recursive step
        curr_key = keys[0]
        if (isinstance(instances[curr_key], list)): 
            # need parameter sweeping
            for each_parameter in instances[curr_key]:
                # replace the list with a single value
                new_instance[curr_key] = each_parameter
                # proceed with the other parameters in keys
                parameter_sweeping(instances, new_instance, keys[1:])
        else: # no need for parameter sweeping
            new_instance[curr_key] = instances[curr_key]
            parameter_sweeping(instances, new_instance, keys[1:])


def execute_all(config_info):
    """execute all instances based on user's config info and default config info

    Args:
        config_info (dict): configuration information from the yaml file
    """
    all_instances = config_info['instances']
    default = config_info['default']

    for single_instance in all_instances:

        # propagate default config to current instance
        for key in default.keys():
            if key not in single_instance.keys():
                single_instance[key] = default[key]

        keys = list(single_instance.keys())
        new_instance = {} # initially empty
        parameter_sweeping(single_instance, new_instance, keys)


if __name__ == "__main__":
    # load configuration information (only once)
    config_info = load_configuration()
    data_dir = config_info['data_dir']
    execute_all(config_info)