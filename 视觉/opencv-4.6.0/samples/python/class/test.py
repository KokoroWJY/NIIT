import cv2
img=cv2.imread('img/lena.jpg',cv2.IMREAD_REDUCED_COLOR_2)
# img1=cv2.blur(img,(3,3))
# img2=cv2.blur(img,(5,5))
# cv2.imshow('image',img)
# cv2.imshow('image1',img1)
img1=cv2.boxFilter(img,-1,(3,3),normalize=True)
img2=cv2.boxFilter(img,-1,(3,3),normalize=False)
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
# cv2.imshow('image2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy

# img = cv2.imread('img/girl.jpg', cv2.IMREAD_REDUCED_COLOR_2)
# cv2.imshow("img", img)
# img2 = cv2.bilateralFilter(img,20,100,100)
# cv2.imshow('imgBlur', img2)
# cv2.waitKey(0)

