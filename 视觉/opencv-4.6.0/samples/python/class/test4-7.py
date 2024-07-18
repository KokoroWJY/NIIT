#test4-7.py:手工实现高斯滤波
import cv2
import numpy as np

def GaussianFilter(img,K_size=3,sigma=1):
    h,w,c = img.shape

    # 零填充
    pad = K_size//2
    out = np.zeros((h + 2*pad,w + 2*pad,c),dtype=np.float)
    out[pad:pad+h,pad:pad+w] = img.copy().astype(np.float)
    
    # 定义滤波核
    K = np.zeros((K_size,K_size),dtype=np.float)
    
    for x in range(-pad,-pad+K_size):
        for y in range(-pad,-pad+K_size):
            K[y+pad,x+pad] = np.exp(-(x**2+y**2)/(2*(sigma**2)))
    K /= (sigma*np.sqrt(2*np.pi))
    K /=  K.sum()
    
    # 卷积的过程
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[pad+y,pad+x,ci] = np.sum(K*tmp[y:y+K_size,x:x+K_size,ci])
    
    out = out[pad:pad+h,pad:pad+w].astype(np.uint8)
    
    return out


# 读取图像
img1 = cv2.imread('img/noise_Guass_img.png')
# 高斯滤波
img2 = GaussianFilter(img1,3,1)
cv2.imshow('original',img1)
cv2.imshow('result',img2)
cv2.waitKey()
cv2.destroyAllWindows()

