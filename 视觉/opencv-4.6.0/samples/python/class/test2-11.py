#test2-11.py：使用cv2.split()函数拆分通道
import cv2
import numpy

#创建一个240*320黑色图像
img=numpy.zeros((240,320,3),dtype=numpy.uint8) 

cv2.imshow('img',img)           #显示原图像

b,g,r = cv2.split(img)          #按通道拆分图像
b[0:239,0:319]=255                    
r[0:239,0:319]=255
g[0:239,0:319]=100

cv2.imshow('img_B',b)          #显示B通道图像
cv2.imshow('img_G',g)          #显示G通道图像
cv2.imshow('img_R',r)          #显示R通道图像


new_img= cv2.merge((b,g,r))
cv2.imshow('new_img',new_img)          #显示合并后的图像

cv2.waitKey(0)