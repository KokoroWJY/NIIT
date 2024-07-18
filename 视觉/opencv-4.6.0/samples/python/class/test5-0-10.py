import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img/clahe.jpg")
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
cv2.imshow('original', img)
plt.hist(img.ravel(), 256)
plt.show()


# img = cv2.imread("lena.jpg", cv2.IMREAD_REDUCED_GRAYSCALE_2)
# thresh, new_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
# print(thresh)

# cv2.imshow('img', img)
# cv2.imshow("new_img", new_img)
# cv2.waitKey()
