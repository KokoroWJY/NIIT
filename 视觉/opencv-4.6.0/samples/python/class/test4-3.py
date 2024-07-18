#test4-3.py:自定义一个卷积核
import cv2
import numpy as np

o=cv2.imread('flower2.jpg')
#设置3×3卷积核
kernel=np.zeros((3,3),np.float32)
kernel[1][1]=5 
kernel[0][1]=-1
kernel[1][0]=-1
kernel[1][2]=-1
kernel[2][1]=-1
print(kernel)    

#src 输入图像的Mat对象
#ddepth 输出图像深度，值为-1则表示与输入图像一致
#kernel 自定义的卷积核
r=cv2.filter2D(o,-1,kernel) 
print(o.shape)
print(r.shape)    
cv2.imshow('original',o)
cv2.imshow('result',r)
cv2.waitKey()
cv2.destroyAllWindows()