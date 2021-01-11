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
    video_name = single_instance['video_name']
    low_qp = single_instance['low_qp']
    high_qp = single_instance['high_qp']
    low_res = single_instance['low_resolution']
    high_res = single_instance['high_resolution']
    reduce_low = single_instance['reduce_low']
    original_images_dir = os.path.join(data_dir, video_name, 'src')
    result_direc = os.path.join("results", video_name)
    low_results_file = f'{video_name}_mpeg_{low_res}_{low_qp}'
    if reduce_low:
        low_results_file = f"{low_results_file}_reduce-low"

    single_instance['high_images_path'] = f'{original_images_dir}'
    single_instance['outfile'] = os.path.join(result_direc, 'stats')
    single_instance['out_cost_file'] = os.path.join(result_direc, 'costs')
    single_instance['ground_truth'] = os.path.join(result_direc, f'{video_name}_gt')
    single_instance['low_results_path'] = os.path.join(result_direc, low_results_file)

    # branching based on baselines
    if baseline == 'gt':
        # skip if result file already exists
        result_file_name = f"{video_name}_gt"
        if single_instance['overwrite'] == False and \
            os.path.exists(os.path.join(result_direc, result_file_name)):
            print(f"Skipping {result_file_name}")
        # otherwise, start execution
        else:
            single_instance['video_name'] = os.path.join(result_direc, result_file_name)
            single_instance['ground_truth'] = False
            subprocess.run(['python', '../play_video.py', 
                            yaml.dump(single_instance)])

    # assume we are running emulation
    elif baseline == 'mpeg':
        # skip if result file already exists
        result_file_name = low_results_file
        
        if single_instance['overwrite'] == False and \
            os.path.exists(os.path.join(result_direc, result_file_name)):
            print(f"Skipping {result_file_name}")
        else:
            single_instance['video_name'] = os.path.join(result_direc, result_file_name)
            subprocess.run(['python', '../play_video.py',
                            yaml.dump(single_instance)])

    elif baseline == 'dds':
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
        if reduce_low:
            result_file_name = f"{result_file_name}_reduce-low"
        if single_instance['overwrite'] == False and \
            os.path.exists(os.path.join(result_direc, result_file_name)):
            print(f"Skipping {result_file_name}")
        else:
            single_instance['video_name'] = os.path.join(result_direc, result_file_name)

            if single_instance["mode"] == 'implementation':
                assert single_instance['hname'] != False, "Must provide the server address for implementation, abort."
                # single_instance['hname'] = '127.0.0.1:5001'
                
            subprocess.run(['python', '../play_video.py',
                                yaml.dump(single_instance)])
    
    elif baseline == 'streamBpack':
        # unpacking
        rpn_enlarge_ratio = single_instance['rpn_enlarge_ratio']
        batch_size = single_instance['batch_size']
        prune_score = single_instance['prune_score']
        objfilter_iou = single_instance['objfilter_iou']
        size_obj = single_instance['size_obj']
        dds_result_file_name = (f"{video_name}_dds_{low_res}_{high_res}_{low_qp}_{high_qp}_"
                            f"{rpn_enlarge_ratio}_twosides_batch_{batch_size}_"
                            f"{prune_score}_{objfilter_iou}_{size_obj}")
        if reduce_low:
            dds_result_file_name = f"{dds_result_file_name}_reduce-low"

        context_padding_type = single_instance['context_padding_type']
        context_val = single_instance['context_val']
        blank_padding_type = single_instance['blank_padding_type']
        blank_val = single_instance['blank_val']
        intersect_iou = single_instance['intersect_iou']
        merge_iou = single_instance['merge_iou']
        resize_method = single_instance['resize_method']
        filter_method = single_instance['filter_method']
        # skip if result file already exists
        # You could customize the way to serialize the parameters into filename by yourself
        result_file_name = (f'{video_name}_streamBpack_{context_padding_type}_{context_val}_'
                            f'{blank_padding_type}_{blank_val}_inter_{intersect_iou}_merge_{merge_iou}_'
                            f'{resize_method}_filter_{filter_method}')
        if reduce_low:
            result_file_name = f"{result_file_name}_reduce-low"
        if single_instance['overwrite'] == False and \
            os.path.exists(os.path.join(result_direc, f"{result_file_name}-high")):
            print(f"Skipping {result_file_name}")
        else:
            single_instance['video_name'] = os.path.join(result_direc, result_file_name)
            single_instance['high_images_path'] = f'{original_images_dir}'
            single_instance['outfile'] = os.path.join(result_direc, 'stats')
            single_instance['out_cost_file'] = os.path.join(result_direc, 'costs')           
            single_instance['req_regions_fname'] = os.path.join(result_direc, \
                f'{dds_result_file_name}-req_regions')
            single_instance['filtered_req_regions_fname'] = os.path.join(result_direc, \
                f'{dds_result_file_name}-filtered_req_regions')

            if single_instance["mode"] == 'implementation':
                assert single_instance['hname'] != False, "Must provide the server address for implementation, abort."
                # single_instance['hname'] = '127.0.0.1:5001'
                
            subprocess.run(['python', '../play_video.py',
                                yaml.dump(single_instance)])
                                
    elif baseline == "onlyObjects":
        result_file_name = f"{video_name}_onlyObjects"
        if single_instance['overwrite'] == False and \
            os.path.exists(os.path.join(result_direc, result_file_name)):
            print(f"Skipping {result_file_name}")
        # otherwise, start execution
        else:
            single_instance['video_name'] = os.path.join(result_direc, result_file_name)

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