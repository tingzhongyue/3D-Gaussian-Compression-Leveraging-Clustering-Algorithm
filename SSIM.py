
import cv2
from skimage.metrics import structural_similarity as ssim


def calculate_ssim(img1, img2):
    # 计算 SSIM
    ssim_score = ssim(img1, img2, multichannel=True)

    return ssim_score


# 示例用法
img1 = cv2.imread('./data/output/test/ours_30000/gt/00000.png')
img2 = cv2.imread('./data/output/test/ours_30000/renders/00000.png')
print(img2.shape)
score = calculate_ssim(img1, img2)
print("SSIM score:", score)

