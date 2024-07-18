# test4-2.py 给图片添加椒盐噪声
# 椒指的是黑色的噪点(0,0,0) 
# 盐指的是白色的噪点(255,255,255)
import cv2
import numpy as np
#读取图片
img = cv2.imread("demo.png")
#设置添加椒盐噪声的数目比例
s_vs_p = 0.5
#设置添加噪声图像像素的数目
amount = 0.04
noisy_img = np.copy(img)
#添加salt噪声
num_salt = np.ceil(amount * img.size * s_vs_p)
#设置添加噪声的坐标位置
coords = [np.random.randint(0,i - 1, int(num_salt)) for i in img.shape]
noisy_img[coords[0],coords[1],:] = [255,255,255]
#添加pepper噪声
num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
#设置添加噪声的坐标位置
coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in img.shape]
noisy_img[coords[0],coords[1],:] = [0,0,0]
#保存图片
cv2.imwrite("noisy_ps_img.png",noisy_img)
cv2.imshow("original_img", img)
cv2.imshow("noise_img.png", noisy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()