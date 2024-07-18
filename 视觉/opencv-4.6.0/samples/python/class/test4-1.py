#test4-1.py 给图片添加高斯噪声
import cv2
import numpy as np

# 添加高斯噪声
# mean：正态分布的均值，对应着这个分布的中心，0表示以Y轴为对称曲线的正态分布
# sigma：正态分布的标准差，对应分布的宽度，sigma越大，正态分布的曲线越矮胖
def add_noise_Guass(img, mean=0, sigma=0.01): 
    img = np.array(img/255, dtype=float)
    noise = np.random.normal(mean, sigma ** 0.5, img.shape)
    cv2.imwrite('img/lena.jpg', noise*255) 
    #给图片添加高斯噪声
    out_img = img + noise
    if out_img.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
        out_img = np.clip(out_img, low_clip, 1.0)
        out_img = np.uint8(out_img * 255)
    return out_img
 
#读取图片
img1 = cv2.imread('img/dog.png') 
out_img = add_noise_Guass(img1)  
cv2.imshow("original_img", img1)
cv2.imshow("noisy_img", out_img)
#在添加噪声的过程中，图像被归一化，需要恢复
cv2.imwrite('noise_Guass_img.png', out_img*255) 
noise = cv2.imread('noise_Guass.png') 
cv2.imshow("noisy", noise)
cv2.waitKey(0)
cv2.destroyAllWindows()




