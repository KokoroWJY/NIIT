#test2-9.py:通过数组索引拆分通道

import cv2
img=cv2.imread('lena.jpg',cv2.IMREAD_REDUCED_COLOR_2)  	#读取图像，缩小为原来的1/2
cv2.imshow('lena',img)          #显示原图像
b=img[:,:,0]                    #获得B通道图像
g=img[:,:,1]                    #获得G通道图像
r=img[:,:,2]                    #获得R通道图像
cv2.imshow('lena_B',b)          #显示B通道图像
cv2.imshow('lena_G',g)          #显示G通道图像
cv2.imshow('lena_R',r)          #显示R通道图像
cv2.waitKey(0)