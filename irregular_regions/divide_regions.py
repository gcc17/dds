from dds_utils import (read_results_dict, Region)
from reduce_Bcost.streamB_utils import (normalize_image)
from reduce_Bcost.new_region import (PadShiftRegion)
import cv2 as cv
import numpy as np
import os


def update_cache(arr_c, arr_b, M, N, x):
    """
    Args:
        arr_c ([M*1 array]): How many 1s to the right of column x(N-1~0)
        arr_b ([M*N array]): original array
    """
    for i in range(M):
        if arr_b[i,x]:
            arr_c[i] += 1
        else:
            arr_c[i] = 0
    return arr_c

def rectangle_area(lower_left, upper_right):
    w = upper_right[0] - lower_left[0] + 1
    h = lower_left[1] - upper_right[1] + 1
    return w*h


def find_largest_rectange(arr_b):
    M = arr_b.shape[0]
    N = arr_b.shape[1]
    arr_c = np.zeros(M)
    best_ll = (0, 0)
    best_ur = (-1, -1)

    for x in range(N-1, -1, -1):
        arr_c = update_cache(arr_c, arr_b, M, N, x)
        width = 0
        lower_stack = []
        for y in range(M-1, -1, -1):
            if arr_c[y] > width:
                lower_stack.append((y, width))
                width = arr_c[y]
            elif arr_c[y] < width:
                while len(lower_stack) > 0:
                    (y0, w0) = lower_stack.pop()
                    if width*(y0-y) > rectangle_area(best_ll, best_ur):
                        best_ll = (x, y0)
                        best_ur = (int(x+width-1), int(y+1))
                        # print(best_ll, best_ur)
                    width = w0
                    if arr_c[y] >= width and arr_c[y] != 0:
                        break
                
                # if width != 0:
                #     width = min(arr_c[y], width)
                #     lower_stack.append((y0, width))
                # else:
                #     width = arr_c[y]
                #     lower_stack.append((y0, width))
                width = arr_c[y]
                if width != 0:
                    lower_stack.append((y0,0))

                
        while len(lower_stack) > 0:
            # import ipdb; ipdb.set_trace()
            (y0, w0) = lower_stack.pop()
            if width*(y0+1) > rectangle_area(best_ll, best_ur):
                best_ll = (x,y0)
                best_ur = (int(x+width-1), 0)
                # print("----", best_ll, best_ur)
            width = w0
        
    return best_ll, best_ur, rectangle_area(best_ll, best_ur)


def divide_single(arr_b, expand_ratio, src_frame=None, save_images_direc=None,
        fname=None, box_color=(0,0,255)):
    all_rects = []
    tmp_arr_b = arr_b.copy()
    while tmp_arr_b.sum() > 0:
        best_ll, best_ur, _ = find_largest_rectange(tmp_arr_b)
        x0 = best_ll[0]
        y0 = best_ur[1]
        x1 = best_ur[0]
        y1 = best_ll[1]
        all_rects.append((x0, y0, x1, y1))
        print(x0, y0, x1, y1)
        tmp_arr_b[y0:y1+1, x0:x1+1] = 0
        if src_frame is not None:
            cv.rectangle(src_frame, (x0, y0), (x1, y1), box_color, 2)
    if src_frame is not None:
        if save_images_direc is None:
            cv.imshow("Divide into rectangles", src_frame)
            key = cv.waitKey()
            if key & 0xFF == ord("q"):
                cv.destroyAllWindows()
        else:
            os.makedirs(save_images_direc, exist_ok=True)
            cv.imwrite(os.path.join(save_images_direc, fname), src_frame,
                   [cv.IMWRITE_PNG_COMPRESSION, 0])


