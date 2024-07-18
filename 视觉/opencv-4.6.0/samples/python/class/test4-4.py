#test4-4.py：2D卷积
import numpy as np
import cv2
img=cv2.imread('img/flower.jpg',cv2.IMREAD_REDUCED_COLOR_2)
#自定义卷积核
k1=np.ones((3,3),np.float32)/9
k2=np.ones((5,5),np.float32)/25
cv2.imshow('img',img)
img1=cv2.filter2D(img,-1,k1)
cv2.imshow('imgK1',img1)
img2=cv2.filter2D(img,-1,k2)
cv2.imshow('imgK2',img2)
cv2.waitKey(0)
