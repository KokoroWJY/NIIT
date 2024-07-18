import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1 读取图像
imgBGR = cv.imread('lena.jpg')
imgRGB = cv.cvtColor(imgBGR, cv.COLOR_BGR2RGB)

# 2创建蒙版
# ①建立与原图一样大小的mask图像，并将所有像素初始化为0，因此全图成了一张全黑色图。
mask = np.zeros(imgRGB.shape[:2], np.uint8)
# ②将mask图中的roi区域的所有像素值设置为255,也就是整个roi区域变成了白色
mask[400:650, 200:500]=255

# 3掩模
#mask图中，感兴趣的区域是白色的255，二进制表示为11111111
#mask图中，非感兴趣区域是黑色的0，其二进制表示为00000000。
#原图与mask图进行与运算后，得到的结果图只留下原始图感兴趣区域的图像了
#注：add调用时，若无mask参数则返回src1&src2；若存在mask参数，则返回src1 & src2 & mask
masked_img = cv.bitwise_and(imgRGB, imgRGB, mask = mask)

# 4统计掩膜后图像的灰度图
mask_his = cv.calcHist([imgRGB], [0], mask, [256], [1, 256])
# 5图像展示
fig, axes=plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
axes[0, 0].imshow(imgRGB)
axes[0, 0].set_title("原图")
axes[0, 1].imshow(mask, cmap=plt.cm.gray)
axes[0, 1].set_title("蒙版数据")
axes[1, 0].imshow(masked_img)
axes[1, 0].set_title("掩膜后数据")
axes[1, 1].plot(mask_his)
axes[1, 1].grid()
axes[1, 1].set_title("灰度直方图")
plt.show()
