import cv2
import numpy as np

img1 = cv2.imread('map1.png')
img2 = cv2.imread('map2.png')
print(img1.shape)
# 灰度化
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 创建SIFI对象
orb = cv2.ORB_create()

# 返回关键点和描述子
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# 创建匹配器
bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
# 执行匹配
ms = bf.match(des1, des2)
# 按照距离排序
ms = sorted(ms, key=lambda x: x.distance)
matchesMask = None

if len(ms) > 10:  # 在有一定数量的匹配结果后，才计算查询图像在训练图像中的位置。
    # 计算查询图像匹配结果的坐标
    querypts = np.float32([kp1[m.queryIdx].pt for m in ms]).reshape(-1, 1, 2)
    # 计算训练图像匹配结果的坐标
    trainpts = np.float32([kp2[m.queryIdx].pt for m in ms]).reshape(-1, 1, 2)
    # 执行查询图像和训练图像的透视变换
    retv, mask = cv2.findHomography(querypts, trainpts, cv2.RANSAC)
    # 计算最佳结果的掩膜，用于绘制匹配的结果
    matchesMask = mask.ravel().tolist()
    h, w, args = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    # 执行向量的透视矩阵变换，获得查询图像在训练图像中的位置
    dst = cv2.perspectiveTransform(pts, retv)
    # 用白色矩形在训练图像中绘制查询图像的范围
    img2 = cv2.polylines(img2, [np.int32(dst)], True, (255, 255, 255), 5)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, ms, None, matchColor=(0, 255, 0), singlePointColor=None,
                           matchesMask=matchesMask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.waitKey()
#
# #创建暴力匹配器  FLANN
# index_params=dict(algorithm=1,trees=5)
# search_params=dict(check=50)
# flann=cv2.FlannBasedMatcher(index_params,search_params)
#
#
# #特性怕匹配
# match=flann.knnMatch(des1,des2,k=2)
#
# good=[]
# #筛选合适的算子
# for i ,(m,n) in enumerate(match):
#     if m.distance <0.7*n.distance:
#         good.append(m)
# #绘制匹配结果
# result=cv2.drawMatchesKnn(img1,kp1,img2,kp2,[good],None)
#
# cv2.imshow('img1',img1)
# cv2.imshow('img2',img2)
# cv2.imshow('result',result)
# cv2.waitKey()
