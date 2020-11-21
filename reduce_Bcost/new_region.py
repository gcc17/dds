import re
import os


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

