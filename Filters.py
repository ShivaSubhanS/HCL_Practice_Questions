#average,gaussian,median,bilateral
import cv2 as cv
x=cv.imread("img.jpg")
a=cv.blur(x,(5,5))
b=cv.GaussianBlur(x,(5,5),0)
c=cv.medianBlur(x,5)
d=cv.bilateralFilter(x,9,75,75)
cv.imshow("Average",a)
cv.imshow("Gaussian",b)
cv.imshow("Median",c)
cv.imshow("Bilateral",d)
cv.waitKey(0)
cv.destroyAllWindows()
