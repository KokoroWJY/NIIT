import cv2

cv2.__version__

imgFile = 'img/1.png'
img1 = cv2.imread(imgFile, flags=0)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img1)
cv2.waitKey(0)
cv2.destroyWindow()

# import numpy as np
#
#
# def sigmoid(x):
#     return 1.0 / (1 + np.exp(-x)) # 以e为底的指数
#
#
# y = sigmoid(6)
# print(y)
