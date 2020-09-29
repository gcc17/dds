import cv2 as cv
import os


def compare_two_image_diff(image1_path=None, image2_path=None, image1=None, image2=None,
        template_method=cv.TM_CCOEFF_NORMED):
    if image1_path and image2_path:
        # read gray image (one channel)
        image1 = cv.imread(image1_path, 0)
        image2 = cv.imread(image2_path, 0)
    elif image1 is None or image2 is None:
        return 10000
    # calculate histogram
    image1_hist = cv.calcHist([image1], [0], None, [256], [0, 256])
    image2_hist = cv.calcHist([image2], [0], None, [256], [0, 256])

    # histogram distance
    img_hist_dist = cv.compareHist(image1_hist, image2_hist, cv.HISTCMP_BHATTACHARYYA)
    # print(f'image histogram Ba-distance: {img_hist_dist}')
    # template matching
    max_w = max(image1.shape[1], image2.shape[1])
    max_h = max(image1.shape[0], image2.shape[0])
    image1 = cv.resize(image1, (max_w, max_h), interpolation=cv.INTER_CUBIC)
    # matchTemplate(image, template, method)
    res = cv.matchTemplate(image1, image2, template_method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    if template_method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        img_match_val = min_val
    else:
        img_match_val = max_val
    # print(f'image template matching value: {img_match_val}')
    img_template_diff = 1 - img_match_val

    # taking only 10% of histogram diff, since it's less accurate than template method
    image_diff = img_hist_dist/10 + img_template_diff
    return image_diff

# 2-4: image histogram Ba-distance: 0.06481135245807323, image template matching value: 0.8631677627563477
### 0.14331337248945966
# 1-3: image histogram Ba-distance: 0.06481135245807323, image template matching value: 0.8631677627563477
### 0.14331337248945966
# 6-4: image histogram Ba-distance: 0.24023210633075207, image template matching value: 0.5453671216964722
### 0.47865608893660305
# 6-1: image histogram Ba-distance: 0.25212744198261755, image template matching value: 0.5655974745750427
### 0.459615269623219
# 1-2: image histogram Ba-distance: 0.29147331063709475, image template matching value: 0.10788745433092117
### 0.9212598767327883


def track_similar_region(merged_new_regions_dict, tmp_regions_direc, max_diff=0.4):
    track_new_regions_dict = {}
    track_new_regions_gray_dict = {}
    track_new_regions_contain_dict = {}
    for merged_region_id in merged_new_regions_dict.keys():
        cur_merged_new_region = merged_new_regions_dict[merged_region_id]
        cur_path = os.path.join(tmp_regions_direc, f"region-{merged_region_id}.png")
        cur_gray_image = cv.imread(cur_path, 0)

        if cur_merged_new_region.original_region.fid == 0:
            track_new_regions_dict[merged_region_id] = cur_merged_new_region    
            track_new_regions_gray_dict[merged_region_id] = cur_gray_image
            track_new_regions_contain_dict[merged_region_id] = []
        else:
            find_similar = False
            for track_id, track_gray_image in track_new_regions_gray_dict.items():
                if compare_two_image_diff(image1=track_gray_image, image2=cur_gray_image) < max_diff:
                    track_new_regions_contain_dict[track_id].append(merged_region_id)
                    find_similar = True
                    break
            if not find_similar:
                track_new_regions_dict[merged_region_id] = cur_merged_new_region    
                track_new_regions_gray_dict[merged_region_id] = cur_gray_image
                track_new_regions_contain_dict[merged_region_id] = []
                    
        
    return track_new_regions_dict, track_new_regions_contain_dict



        


