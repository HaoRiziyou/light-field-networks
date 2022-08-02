"""
    compare single pair of ground truth and reconstruction with PSNR and SSIM
"""
import cv2
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity

img1 = cv2.imread('./0000_4a55.png')
img2 = cv2.imread('./4a55_1.png')



img3 = cv2.imread('./0000_5aca.png')
img4 = cv2.imread('./5aca_1.png')


img5 = cv2.imread('./0000_2086.png')

img6 = cv2.imread('./2086_1.png')









image_gt_list = []
image_noise_list =[]
image_gt_list.append(img1)
image_gt_list.append(img3)
image_gt_list.append(img5)



image_noise_list.append(img2)
image_noise_list.append(img4)
image_noise_list.append(img6)

#for x,y in zip(image_gt_list, image_noise_list):
#  
#  PSNR = peak_signal_noise_ratio(x, y)
#  SSIM = structural_similarity(x, y, multichannel=True)
#  
#  print('SSIM: ', SSIM)
#
#  print('PSNR: ', PSNR)
#

x = cv2.imread('./000001_gt_rgb_0_maybeinlier.png')
y = cv2.imread('./000001_final.png')

PSNR = peak_signal_noise_ratio(x,y)
SSIM = structural_similarity(x,y, multichannel=True)
print('SSIM: ', SSIM)

print('PSNR: ', PSNR)
  