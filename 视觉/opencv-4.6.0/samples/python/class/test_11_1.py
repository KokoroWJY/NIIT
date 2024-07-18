from turtle import hideturtle
import cv2
import numpy as np

# img = cv2.imread('img/j.png')
# cv2.imshow('img', img)
# kernel = np.ones((5,5), np.uint8)
# dst = cv2.erode(img, kernel, iterations=1)
# cv2.imshow('dst', dst)
# cv2.waitKey(0)



# img=cv2.imread('img/j.png')
# cv2.imshow('img',img)
# kernel=np.ones((3,3),np.uint8)
# op=cv2.MORPH_OPEN
# dst=cv2.morphologyEx(img,op,kernel,iterations=5)
# cv2.imshow('dst',dst)
# cv2.waitKey()


# img=cv2.imread('img/j.png')
# cv2.imshow('img',img)
# kernel = np.ones((3,3),np.uint8)    #卷积核越大，图片腐蚀越严重
# dst=cv2.dilate(img,kernel,iterations=1)
# cv2.imshow('dst',dst)
# cv2.waitKey()

# img = cv2.imread('img/j.png')
# cv2.imshow('img', img)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# op = cv2.MORPH_GRADIENT
# dst = cv2.morphologyEx(img, op, kernel, iterations=1)
# cv2.imshow('dst', dst)
# cv2.waitKey(0)

# erode_dst = cv2.erode(img, kernel, iterations=1)
# subtract_dst = cv2.subtract(img, erode_dst)

# img = cv2.imread('img/new/AB.png')
# cv2.imshow('img', img)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# op = cv2.MORPH_OPEN
# dst = cv2.morphologyEx(img, op, kernel, iterations=5)
# cv2.imshow('dst', dst)
# cv2.waitKey(0)


img=cv2.imread('img/new/findContours1.png',cv2.IMREAD_REDUCED_GRAYSCALE_2)

cv2.imshow('img',img)

ret,binary=cv2.threshold(img,127,255,cv2.THRESH_BINARY)

contours,hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

print(contours,hierarchy)
newImg = np.zeros(img.shape, np.uint8) + 255
cv2.drawContours(newImg, contours, -1, (0,0,255), 2)
cv2.imshow('newImg', newImg)
cv2.waitKey()

cv2.contourArea(contours)
ret = cv2.arcLength(contours, closed=1)




