import cv2
import numpy as np
import re
import math


def read_calib(file_path):
    info = {}

    with open(file_path, 'r') as f:
        content = f.read()

    pattern = r'(\w+)\s*=\s*(\[[^\]]+\]|[\d\.]+)' # pattern for matching key-value pairs
    matches = re.findall(pattern, content)

    for key, value in matches:
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            nums = [float(x) for x in value[1:-1].split()]
            info[key] = [nums[i:i+3] for i in range(0, 9, 3)]
        elif '.' in value:
            info[key] = float(value)
        else:
            info[key] = int(value)

    return info

def disparity(img_left, img_right, ndisp):
    num_disparities = math.ceil(ndisp / 16.0) * 16

    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=15)

    raw_disparity = stereo.compute(img_left, img_right)

    disparity = raw_disparity.astype(np.float32) / 16.0

    return disparity

def save_noramlize_disparity(disparity, output_path):
    disp_vis = np.clip(disparity, 0, None)
    disp_normalized = cv2.normalize(disp_vis, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    heatmap = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, heatmap)
    
    return heatmap

def get_pixel_depth(disparity_map, calib_info, dataset_label="Dataset"):
    h, w = disparity_map.shape
    px_x, px_y = w // 2, h // 2
    
    disp = disparity_map[px_y, px_x]
    
    if disp <= 0 or np.isnan(disp):
        depth = None
        print(f"{dataset_label} - Pixel ({px_x}, {px_y}): Disparity = {disp:.2f} px, Depth = Invalid/Out of Range")
    else:
        focal_length = calib_info['cam0'][0][0]
        baseline = calib_info['baseline']
        doffs = calib_info['doffs']

        depth = (baseline * focal_length) / (disp + doffs)
        print(f"{dataset_label} - Pixel ({px_x}, {px_y}): Disparity = {disp:.2f} px, Depth = {depth:.2f} mm")

    return disp, depth


def main():
    img0l = cv2.cvtColor(cv2.imread('dataset/img0/im0.png'),cv2.COLOR_BGR2GRAY)
    img0r = cv2.cvtColor(cv2.imread('dataset/img0/im1.png'),cv2.COLOR_BGR2GRAY)

    img1l = cv2.cvtColor(cv2.imread('dataset/img1/im0.png'),cv2.COLOR_BGR2GRAY)
    img1r = cv2.cvtColor(cv2.imread('dataset/img1/im1.png'),cv2.COLOR_BGR2GRAY)

    info0 = read_calib('dataset/img0/calib.txt')
    info1 = read_calib('dataset/img1/calib.txt')

    disparity0 = disparity(img0l, img0r, info0['ndisp'])
    disparity1 = disparity(img1l, img1r, info1['ndisp'])

    save_noramlize_disparity(disparity0, 'disparity0.png')
    save_noramlize_disparity(disparity1, 'disparity1.png')

    get_pixel_depth(disparity0, info0, "Dataset 0")
    get_pixel_depth(disparity1, info1, "Dataset 1")



if __name__ == '__main__':
    main()