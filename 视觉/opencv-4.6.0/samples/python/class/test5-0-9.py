import cv2
import math
import numpy as np

img = cv2.imread('img/test3.jpg',cv2.IMREAD_REDUCED_COLOR_2)
log_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        log_img[i, j, 0] = math.log(1 + img[i, j, 0])
        log_img[i, j, 1] = math.log(1 + img[i, j, 1])
        log_img[i, j, 2] = math.log(1 + img[i, j, 2])
        
#归一化到0~255  
cv2.normalize(log_img, log_img, 0, 255, cv2.NORM_MINMAX)
#转换成8bit图像显示
log_img = cv2.convertScaleAbs(log_img)

cv2.imshow('img/test3.jpg', img)
cv2.imshow('log transform', log_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
