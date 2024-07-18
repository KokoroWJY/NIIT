# 选择一幅彩色图像，完成图像的平移，缩放，旋转，透视等操作。
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

img = cv2.imread('img/catpicture.jpg', cv2.IMREAD_REDUCED_COLOR_2)
cv2.imshow('1', img)
print(img.shape)

# 平移
height = img.shape[0]
width = img.shape[1]
dsize = (width, height)  # 变换后图像的大小(图像框的大小)
m = np.float32([[1, 0, 100], [0, 1, 50]])  # 变换矩阵
# src – 输入图像。   M – 变换矩阵。  dsize – 输出图像的大小
img1 = cv2.warpAffine(img, m, dsize)
cv2.imshow('translate', img1)

# 缩放
m = np.float32([[0.5, 0, 0], [0, 0.8, 0]])  # 变换矩阵
img2 = cv2.warpAffine(img, m, dsize)
cv2.imshow('zoom', img2)

# 旋转  180度
img3 = cv2.rotate(img, cv2.ROTATE_180)
cv2.imshow('rotate', img3)

# 透视
new_img = cv2.imread('img/lena.jpg')
height = new_img.shape[0]
width = new_img.shape[1]
dsize = (width, height)  # 变换后图像的大小(图像框的大小)
# src=np.float32([[90,90],[835,155],[935,885],[205,825]])
src = np.float32([[90, 90], [835, 155], [935, 885], [205, 825]])  # 取原图中的四个点（自己定义）
dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])  # 原图像中4个点在转换后的目标图像中的对应坐标
m = cv2.getPerspectiveTransform(src, dst)  # 计算透视变换的转换矩阵    矩阵大小是3*3    得出一个矩阵
img4 = cv2.warpPerspective(new_img, m, dsize)
img5 = cv2.resize(img4, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
cv2.imshow('toushi', img5)

cv2.waitKey(0)
