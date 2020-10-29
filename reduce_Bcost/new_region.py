import re
import os

class ShiftRegion:
    def __init__(self, original_region, region_id, 
            abs_neigh_x, abs_neigh_y, abs_neigh_w, abs_neigh_h, 
            abs_total_w, abs_total_h, blank_pad_w, blank_pad_h,
            relative_neigh_x, relative_neigh_y, 
            abs_resize_ori_w, abs_resize_ori_h,
            shift_x=0, shift_y=0,
            fx=1, fy=1):
        # shift_x: float, shift of the region, [0,1]
        # abs_neigh_x: left top of the neigh-padded region 
        # region_id: req_region id
        self.original_region = original_region
        self.region_id = region_id
        self.abs_neigh_x = abs_neigh_x
        self.abs_neigh_y = abs_neigh_y
        self.abs_neigh_w = abs_neigh_w
        self.abs_neigh_h = abs_neigh_h
        self.abs_total_w = abs_total_w
        self.abs_total_h = abs_total_h
        self.blank_pad_w = blank_pad_w
        self.blank_pad_h = blank_pad_h
        self.relative_neigh_x = relative_neigh_x
        self.relative_neigh_y = relative_neigh_y
        self.abs_resize_ori_w = abs_resize_ori_w
        self.abs_resize_ori_h = abs_resize_ori_h
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.fx = fx
        self.fy = fy
        self.gt_region_list = []


class PadShiftRegion:
    def __init__(self, original_region, region_id, x, y, w, h, fx=1, fy=1, shift_x=0, shift_y=0):
        self.original_region = original_region
        self.region_id = region_id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.fx = fx
        self.fy = fy
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.blank_x = -1
        self.blank_y = -1
    
    def write_new_region_txt(self, fname):
        with open(fname, 'a') as f:
            str_to_write = (f'{self.original_region.fid},{self.original_region.x},{self.original_region.y},'
                            f'{self.original_region.w},{self.original_region.h},'
                            f'{self.region_id},{self.x},{self.y},{self.w},{self.h},{self.fx},{self.fy},'
                            f'{self.shift_x},{self.shift_y}\n')
            f.write(str_to_write)

    def write_new_region_csv(self, fname):
        with open(fname, 'a') as f:
            csv_writer = csv.writer(f)
            row = [self.original_region.fid, self.original_region.x, self.original_region.y,
                    self.original_region.w, self.original_region.h,
                    self.region_id, self.x, self.y, self.w, self.h, self.fx, self.fy, 
                    self.shift_x, self.shift_y]
            csv_writer.writerow(row)

    def write(self, fname):
        if re.match(r"\w+[.]csv\Z", fname):
            self.write_new_region_csv(fname)
        else:
            self.write_new_region_txt(fname)

