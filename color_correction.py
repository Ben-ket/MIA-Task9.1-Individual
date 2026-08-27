import cv2
import numpy as np

def gray_world_white_balance(img):
    # adjusting channels assuming average scene reflectance is gray 
    result = img.astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])

    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    # scale channels to balance back red & blue relative to green
    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / (avg_b + 1e-5)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / (avg_g + 1e-5)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / (avg_r + 1e-5)), 0, 255)

    return result.astype(np.uint8)


def apply_clahe_lab(img):
    # enhances image contrast using CLAHE on the Lightness channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def main():
    img = cv2.imread('dataset/side_quest_2.png')

    output_path = 'side_quest_2_corrected.png'

    # correct heavy cyan/green tint using white Balance
    balanced = gray_world_white_balance(img)

    # restoring contrast using local adaptive equalization
    corrected = apply_clahe_lab(balanced)

    cv2.imwrite(output_path, corrected)

if __name__ == "__main__":
    main()