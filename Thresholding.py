#global,adaptive,Otsu
import cv2 as cv
x=cv.imread("img.jpg",0)
_,a=cv.threshold(x,127,255,cv.THRESH_BINARY)
b=cv.adaptiveThreshold(x,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
c=cv.adaptiveThreshold(x,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
_,d=cv.threshold(x,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imshow("Global",a)
cv.imshow("AdaptiveMean",b)
cv.imshow("AdaptiveGaussian",c)
cv.imshow("Otsu",d)
cv.waitKey(0)
cv.destroyAllWindows()
