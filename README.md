# Liveness Detection Scripts
This repository contains three Python scripts for liveness detection using different techniques:

**Consecutive Frame Liveness Detection**
This script performs liveness detection by examining the variation of pixel values between two consecutive frames. The idea is that a real face will have some variation in pixel values between two consecutive frames due to natural movements and changes in lighting, while a fake face may not have the same variation. The difference between two consecutive frames is computed using the cv2.absdiff function, and the mean value of the difference image is calculated. If the mean difference is above a certain threshold, the person is considered fake. Otherwise, they are considered real.

**Optical Flow Liveness Detection**
This script uses optical flow to distinguish between real 3D objects and fake 2D planes. The more irregular flow pattern of a 3D object is compared to the uniform flow of a 2D plane. The average flow magnitude is calculated using cv2.calcOpticalFlowFarneback, and if the ratio of flow magnitudes in row and column directions is below a threshold, the object is considered real, otherwise fake.

**Fourier Liveness Detection**
This script performs liveness detection by examining the frequency content of the face in the Fourier domain. The idea is that a real face will have certain frequency content in the low-frequency region of the Fourier spectrum, while a fake face may not have the same frequency content. The face ROI is extracted from the frame and its two-dimensional discrete Fourier transform (DFT) is computed. The power spectrum of the DFT is then computed, and the mean power in the low-frequency region is calculated. If the mean power in the low-frequency region is above a certain threshold, the person is considered fake. Otherwise, they are considered real.

**Blink Detection Liveness Detection**
This code performs liveness detection by detecting blinks using the eye aspect ratio. The eye aspect ratio is a measure of the relative distance between the eye landmarks. The basic idea behind blink detection is that a real person will naturally blink their eyes, while a fake face may not have the ability to blink.

Blink detection is performed using the eye aspect ratio, which measures the relative distance between the eye landmarks. If the eye aspect ratio falls below a certain threshold for a certain number of consecutive frames, the person is considered to be blinking. The liveness score is then determined as the inverse of the blink state - if the person is blinking, the score is 0, and if the person is not blinking, the score is 1.



## Requirements
All three scripts require the following Python packages to be installed:
- OpenCV (cv2)
- NumPy

## Usage
To use any of these scripts, simply run them in a Python environment with the required packages installed. You may need to modify the code to adjust the threshold values used for liveness detection, depending on your specific use case.

For example, to run consecutive_frame_liveness_detection.py, simply run the following command in your terminal or command prompt:

```bash
python consecutive_frame_liveness_detection.py
```

## References
The code for these scripts was adapted from various sources, including:

https://github.com/lincolnhard/head-pose-estimation
https://www.pyimagesearch.com/2021/01/11/opencv-liveness-detection-with-python/
https://www.learnopencv.com/fourier-transform-in-opencv/
