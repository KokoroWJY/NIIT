#test2-13.py 
import cv2
img=cv2.imread('bee.jpg')  	            #读取图像
cv2.imshow('BGR',img)  	                #显示图像
img2=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #转换色彩空间为RGB
cv2.imshow('RGB',img2)  	             #显示图像
cv2.waitKey(0)