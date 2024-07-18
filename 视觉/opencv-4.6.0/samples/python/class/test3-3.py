#test2-14.py：图像的加权加法运算
import cv2
img1=cv2.imread('img/lena.jpg',cv2.IMREAD_REDUCED_COLOR_2)  		#读取图像
img2=cv2.imread('img/opencvlog.jpg',cv2.IMREAD_REDUCED_COLOR_2)	#读取图像

#图像宽、高、通道数一样才能进行加权加法运算
print(img1.shape)
print(img1.dtype)
print(img2.shape)
print(img2.dtype)

a = 2
b = 1
g = 0

img3=cv2.addWeighted(img1,a,img2,b,g)

cv2.imshow('lena',img1)          					#显示原图像
cv2.imshow('log',img2)          					#显示原图像
cv2.imshow('lena+log',img3)         				#显示addWeighted()函数运算结果图像

print('img1[126,20]:',img1[126,20])
print('img2[126,20]:',img2[126,20])
print('img3[126,20]:',img3[126,20])

cv2.waitKey(0)
