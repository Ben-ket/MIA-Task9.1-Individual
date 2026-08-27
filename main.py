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
    block_size = 7
    
    # StereoSGBM reduces disparity noise compared to StereoBM
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    raw_disparity = stereo.compute(img_left, img_right)
    return raw_disparity.astype(np.float32) / 16.0

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

# BONUS

def generate_point_cloud(disparity_map, img_bgr, calib_info, max_depth=10000.0):
    f = calib_info['cam0'][0][0]
    cx = calib_info['cam0'][0][2]
    cy = calib_info['cam0'][1][2]
    B = calib_info['baseline']
    doffs = calib_info['doffs']

    h, w = disparity_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    
    valid_mask = (disparity_map > 1.0) & (~np.isnan(disparity_map))

    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    d_valid = disparity_map[valid_mask]

    Z = (B * f) / (d_valid + doffs)
    
    
    depth_mask = (Z > 0) & (Z < max_depth)
    
    Z = Z[depth_mask]
    u_valid = u_valid[depth_mask]
    v_valid = v_valid[depth_mask]

    X = -((u_valid - cx) * Z) / f
    Y = -((v_valid - cy) * Z) / f

    points = np.vstack((X, Y, Z)).T
    colors = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)[valid_mask][depth_mask]

    return points, colors


def write_ply(filename, points, colors):
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    data = np.hstack((points, colors.astype(np.uint32)))

    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.4f %.4f %.4f %d %d %d")


def main():
    img0_bgr = cv2.imread('dataset/img0/im0.png')
    img0l = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)
    img0r = cv2.cvtColor(cv2.imread('dataset/img0/im1.png'), cv2.COLOR_BGR2GRAY)

    img1_bgr = cv2.imread('dataset/img1/im0.png')
    img1l = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    img1r = cv2.cvtColor(cv2.imread('dataset/img1/im1.png'), cv2.COLOR_BGR2GRAY)

    info0 = read_calib('dataset/img0/calib.txt')
    info1 = read_calib('dataset/img1/calib.txt')

    disparity0 = disparity(img0l, img0r, info0['ndisp'])
    disparity1 = disparity(img1l, img1r, info1['ndisp'])

    save_noramlize_disparity(disparity0, 'disparity0.png')
    save_noramlize_disparity(disparity1, 'disparity1.png')

    get_pixel_depth(disparity0, info0, "Dataset 0")
    get_pixel_depth(disparity1, info1, "Dataset 1")

    pts0, col0 = generate_point_cloud(disparity0, img0_bgr, info0)
    write_ply('point_cloud0.ply', pts0, col0)
    
    pts1, col1 = generate_point_cloud(disparity1, img1_bgr, info1)
    write_ply('point_cloud1.ply', pts1, col1)



if __name__ == '__main__':
    main()