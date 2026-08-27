# Side Quest 2: Underwater Color Correction

## What's the Problem?
Taking pictures underwater makes the picture look bad as the water absorbs light.
Red light disappears very quickly underwater, while blue and green light travel much deeper.
Because of this, underwater images lose their contrast and look very cyan tinted.

## How I Fixed It

### 1. White Balance
* **Why use it:** To fix the heavy cyan tint and bring back the weak red channel.
* **How it works:** This algorithm assumes that if you take the average of all pixels in a normal picture, it should equal a neutral gray. The code calculates the average values for the Red, Green, and Blue channels separately. Then, it scales each channel so they balance out, which boosts the red channel back to a normal level.

### 2. Contrasting using CLAHE
* **Why use it:** To fix contrast and bring out small details without messing up the colors.
* **How it works:** Changing brightness directly on RGB images can ruin colors. To avoid this, the code converts the image to **LAB color space**. This separates brightness (`L` channel) from color (`A` and `B` channels). Running **CLAHE** (Contrast Limited Adaptive Histogram Equalization) only on the `L` channel boosts the local contrast and details while keeping the colors natural.

## Files in this Branch
* `dataset/side_quest_2.png`: The original image underwater
* `color_correction.py`: The python script that runs the correction
* `side_quest_2_corrected.png`: The final output image with color correction applied