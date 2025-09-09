# Morphological operations: dilation,erosion,opening,closing
import cv2 as cv
import numpy as np
x=cv.imread("img.jpg",0)
y=np.ones((5,5),np.uint8)
a=cv.dilate(x,y,iterations=1)
b=cv.erode(x,y,iterations=1)
c=cv.morphologyEx(x,cv.MORPH_OPEN,y)
d=cv.morphologyEx(x,cv.MORPH_CLOSE,y)
cv.imshow("Dilation",a)
cv.imshow("Erosion",b)
cv.imshow("Opening",c)
cv.imshow("Closing",d)
cv.waitKey(0)
cv.destroyAllWindows()
