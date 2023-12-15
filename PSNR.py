import cv2
import numpy as np
import math


def psnr(img1, img2):
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr2 = 20 * math.log10(1 / math.sqrt(mse))
    return psnr2




gt = cv2.imread('./data/output/test/ours_30000/gt/00000.png')
img = cv2.imread('./data/output/test/ours_30000/renders/00000.png')

print(psnr(gt, img))