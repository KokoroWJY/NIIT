#test3-4.py：图像位运算——非操作
import cv2
import numpy as np

#创建一张图片
img1=np.zeros((200,200),np.uint8)
img1[50:150,50:150]=255

img2=cv2.bitwise_not(img1)

cv2.imshow('img1',img1)   #显示原图像
cv2.imshow('img2',img2)   #非操作图像

cv2.waitKey(0)