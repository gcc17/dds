class MergeLowRegion:
    def __init__(self, old_x, old_y, old_w, old_h, old_res, old_fid, \
            new_x, new_y, new_w, new_h, new_res):
        self.old_x = old_x
        self.old_y = old_y
        self.old_w = old_w
        self.old_h = old_h
        self.old_res = old_res
        self.old_fid = old_fid
        self.x = new_x
        self.y = new_y
        self.w = new_w
        self.h = new_h
        self.new_res = new_res
