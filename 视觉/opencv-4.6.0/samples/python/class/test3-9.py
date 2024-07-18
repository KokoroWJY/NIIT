#test3-9.py：作业1
from unittest import makeSuite
import cv2
import numpy as np

#①读取图像
src=cv2.imread('lena.jpg')  	 
#②创建logo
#红色正方形，绿色正方形，并且绿色正方形有一小部分压住了红色正方形
logo = np.zeros((200,200,3),np.uint8)
logo[20:120,20:120]=[0,0,255]
logo[80:180,80:180]=[0,255,0]
#创建掩码，与logo大小一致，
mask1 = np.zeros((200,200),np.uint8)
mask1[20:120,20:120]=255
mask1[80:180,80:180]=255
#对mask按位求反
mask2= cv2.bitwise_not(mask1)
#原图像中选择添加logo的区域
roi = src[0:200,0:200]
cv2.imshow('logo',logo) 
cv2.imshow('mask1',mask1)
cv2.imshow('m2',mask2)
cv2.imshow('roi',roi)
#与mask2进行与操作,注意是mask2是单通道的
tmp1 = cv2.bitwise_and(roi,roi,mask=mask2)
tmp2 = cv2.add(tmp1,logo)
#roi为深拷贝，因此需要区域赋值去改变原图像
src[0:200,0:200]=tmp2

cv2.imshow('tmp1',tmp1)
cv2.imshow('tmp2',tmp2)
cv2.imshow('src',src)
cv2.waitKey(0)