# #矩阵检索与赋值
# import numpy as np
# import cv2

# img = np.zeros((480,640,3),np.uint8)

# count=0
# while count < 200:
# #    img [count, 100, 2] = 255 #BGR,2对应R通道
#    #BGR 
#    img [count, 100]=[0,255, 255]
#    count = count + 1

# cv2.imshow('img',img)
# key= cv2.waitKey(0)
# if key==27:
#     cv2.destroyAllWindows()         #关闭所有窗口


#矩阵ROI
import numpy as np
import cv2

img = np.zeros((480,640,3),np.uint8)

#提取范围：y从100到200，x从200到400
roi=img[100:200,200:400] 
roi[:,:] =[255,0,255]
#在已取得的区域，再获取一个小区域
roi[10:90,10:190] =[0,255,255]

cv2.imshow('img',img)
key= cv2.waitKey(0)
if key==27:
    cv2.destroyAllWindows()         #关闭所有窗口