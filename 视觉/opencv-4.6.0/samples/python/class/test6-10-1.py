
import cv2
img = cv2.imread('math.png', cv2.IMREAD_REDUCED_COLOR_8)
img1= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh, new_img = cv2.threshold(img1, 180, 255, cv2.THRESH_BINARY_INV)
print(new_img.shape)
print("\n thresh=", thresh)

cv2.imshow('img', img)
cv2.imshow('gray_img', img1)
cv2.imshow('new_img', new_img)
cv2.waitKey()

