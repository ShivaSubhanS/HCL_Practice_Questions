# Sobel,Canny,Laplacian edge detectors
import cv2 as cv
x=cv.imread("img.jpg",0)
y=cv.Sobel(x,cv.CV_64F,1,0,ksize=5)
z=cv.Sobel(x,cv.CV_64F,0,1,ksize=5)
a=cv.Laplacian(x,cv.CV_64F)
b=cv.Canny(x,100,200)
cv.imshow("SobelX",y)
cv.imshow("SobelY",z)
cv.imshow("Laplacian",a)
cv.imshow("Canny",b)
cv.waitKey(0)
cv.destroyAllWindows()
