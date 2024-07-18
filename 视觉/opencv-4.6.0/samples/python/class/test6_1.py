import cv2
import numpy as np

# img = cv2.imread('img/chess.png', cv2.IMREAD_REDUCED_GRAYSCALE_2)
# kernelx = np.array([[-1,-1,-1], [0,0,0], [1,1,1]], dtype=int)
# kernely = np.array([[-1,0,1], [-1,0,1], [-1,0,1]], dtype=int)
# x = cv2.filter2D(img, cv2.CV_16S, kernelx)
# y = cv2.filter2D(img, cv2.CV_16S, kernely)

# absX = cv2.convertScaleAbs(x)
# absY = cv2.convertScaleAbs(y)
# Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# cv2.imshow('img', img)
# cv2.imshow('Prewitt', Prewitt)

# img = cv2.imread('img/chess.png')
# kernelx = np.array([[-1,-2,-1], [0,0,0], [1,2,1]], dtype=int)
# kernely = np.array([[-1,0,1], [-2,0,2], [-1,0,1]], dtype=int)
# x = cv2.filter2D(img, cv2.CV_16S, kernelx)
# y = cv2.filter2D(img, cv2.CV_16S, kernely)

# dst = cv2.Sobel(img, cv2.CV_16S, 0, 1)

# absX = cv2.convertScaleAbs(x)
# absY = cv2.convertScaleAbs(y)
# Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# cv2.imshow('img', img)
# cv2.imshow('Sobel_y', absY)
# cv2.imshow('Sobel_x', absX)
# cv2.imshow('Sobel', Sobel)
# cv2.waitKey()

# img = cv2.imread("img/moon.png")
# kernel1 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
# kernel2 = np.array([[0,1,0], [1,-4,1],[0,1,0]])

# temp1 = cv2.filter2D(img, cv2.CV_16S, kernel1)
# temp2 = cv2.filter2D(img, cv2.CV_16S, kernel2)

# Laplacian1 = img + temp1
# Laplacian2 = img - temp2

# temp3 = cv2.convertScaleAbs(temp1)
# temp4 = cv2.convertScaleAbs(temp2)
# result1 = cv2.convertScaleAbs(Laplacian1)
# result2 = cv2.convertScaleAbs(Laplacian2)

# cv2.imshow('img', img)
# cv2.imshow('Laplacian+', temp3)
# cv2.imshow('result+', result1)
# cv2.imshow('Laplacian', temp4)
# cv2.imshow('result', result2)
# cv2.waitKey()

# img = cv2.imread("lena.jpg", cv2.IMREAD_REDUCED_COLOR_2)
# cv2.imshow('original', img)
# img2 = cv2.Canny(img, 100, 300)
# cv2.imshow('Canny', img2)
# cv2.waitKey()

img = cv2.imread('img/math.png', cv2.IMREAD_REDUCED_GRAYSCALE_4)
thresh, new_img = cv2.threshold(img,)