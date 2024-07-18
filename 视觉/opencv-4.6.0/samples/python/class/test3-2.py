#test3-2.py：图像减法运算
import cv2
import numpy as np

#读取图像
img1=cv2.imread('img/dog.jpeg',cv2.IMREAD_REDUCED_COLOR_4)  	
print(img1.shape)
print(img1.dtype)

#图像加法运算就是矩阵的加法运算
#加法运算的两张图片必须是相等的 
img2=np.ones(img1.shape,img1.dtype)*100

img3=img1-img2
img4=cv2.subtract(img1,img2)
cv2.imshow('img1',img1)          		#显示原图像
cv2.imshow('img2',img2)          		#显示原图像
cv2.imshow('img3',img3)         	#显示“+”图像
cv2.imshow('img4',img4)     		#显示add()图像
cv2.waitKey(0)

