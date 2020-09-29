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
    def __init__(self, original_region, region_id, x, y, w, h, fx=1, fy=1):
        self.original_region = original_region
        self.region_id = region_id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.fx = fx
        self.fy = fy
        self.abs_w = 0
        self.abs_h = 0
