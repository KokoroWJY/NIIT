#test3-4.py：图像位运算——异或操作
import cv2
import numpy as np

#创建一张图片
img1=np.zeros((200,200),np.uint8)
img2=np.zeros((200,200),np.uint8)

img1[20:120,20:120]=255
img2[80:180,80:180]=255

img3=cv2.bitwise_xor(img1,img2)

cv2.imshow('img1',img1)   #显示原图像1
cv2.imshow('img2',img2)   #显示原图像2
cv2.imshow('img3',img3)   #显示处理后图像

cv2.waitKey(0)