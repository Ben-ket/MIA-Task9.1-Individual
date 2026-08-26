import cv2
import numpy as np
import re
import math


def read_calib(file_path):
    info = {}

    with open(file_path, 'r') as f:
        content = f.read()

    pattern = r'(\w+)\s*=\s*(\[[^\]]+\]|[\d\.]+)'
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
    


def main():
    img0l = cv2.cvtColor(cv2.imread('dataset/img0/im0.png'),cv2.COLOR_BGR2GRAY)
    img0r = cv2.cvtColor(cv2.imread('dataset/img0/im1.png'),cv2.COLOR_BGR2GRAY)

    img1l = cv2.cvtColor(cv2.imread('dataset/img1/im0.png'),cv2.COLOR_BGR2GRAY)
    img1r = cv2.cvtColor(cv2.imread('dataset/img1/im1.png'),cv2.COLOR_BGR2GRAY)

    info0 = read_calib('dataset/img0/calib.txt')
    info1 = read_calib('dataset/img1/calib.txt')

    disparity0 = disparity(img0l, img0r, info0['ndisp'])
    disparity1 = disparity(img1l, img1r, info1['ndisp'])



if __name__ == '__main__':
    main()