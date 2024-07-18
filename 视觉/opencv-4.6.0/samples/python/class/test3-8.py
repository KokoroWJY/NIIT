#test3-8.py：实验1：为人物图像打码
import cv2
src1=cv2.imread('img/lena.jpg',cv2.IMREAD_REDUCED_COLOR_2)  	 #读取图像
cv2.imshow('lena',src1)          #显示原图像
src1[120:140,115:190]=[255,0,0]  #更改像素，眼部打码
cv2.imshow('result',src1)        #显示打码图像
cv2.waitKey(0)