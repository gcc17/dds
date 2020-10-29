from .new_region import (PadShiftRegion)
import os
import ipdb
import cv2 as cv
import numpy as np
import shutil
from math import sqrt
from dds_utils import (Results)
import copy
from collections import OrderedDict


def resize_correspond_regions(
        req_new_regions_dict, merged_new_regions_dict, merged_new_regions_contain_dict, 
        mpeg_regions_result, resize_method
    ):
    # Resizing change merged_new_region wh, req_new_region xywh
    if resize_method == 'resize_no':
        return
    


