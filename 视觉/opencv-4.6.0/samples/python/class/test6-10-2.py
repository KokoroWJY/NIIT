#自适应阈值处理
import cv2
#读取图像，将其转换为单通道灰度图像
#img=cv2.imread('lena.jpg',cv2.IMREAD_REDUCED_GRAYSCALE_2) 
img=cv2.imread('math.png',cv2.IMREAD_REDUCED_GRAYSCALE_4) 
thresh, new_img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
#阈值处理
atmc_img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,10)
atgc_img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,10) 
cv2.imshow('img',img)
cv2.imshow('threshold',new_img)
cv2.imshow('atmc_img',atmc_img)
cv2.imshow('atgc_img',atgc_img)
cv2.waitKey(0)
