from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
#import matplotlib.pyplot as plt
image = cv2.imread('C:/Users/hp/OneDrive/Desktop/pp4.jpg')
blured = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blured = cv2.GaussianBlur(image, (9, 9),cv2.BORDER_DEFAULT)
blured = cv2.medianBlur(image,5)
#thresh = cv2.threshold(blured, 60, 255, cv2.THRESH_BINARY)[1]
edged = cv2.Canny(blured,100,100)
edged = cv2.dilate(edged, None, iterations = 12)
edged = cv2.erode(edged, None, iterations = 12)

#thresh = cv2.threshold(image,60,225,cv2.THRESH_BINARY)[1]
#edged=cv2.Canny(blured,35,125)
contour = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contour = imutils.grab_contours(contour)
(contour,_) = contours.sort_contours(contour)
for (i,c) in enumerate(contour):
    if cv2.contourArea(c) < 100:
        continue
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    cv2.drawContours(image,[box.astype("int")], -1, (0,100,0), 4)
#cv2.circle(image,(x,y),7, (255,255,255),-1)
#cv2.putText(image,"{},{}".format(x,y), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)       
cv2.imshow('image',image)
cv2.waitKey(0)
'''    
contour = np.asarray(contour)
for c in contour:
    m = cv2.moments(c)
    x = int(m['m01']/m['m00'])
    y = int (m['m10']/m['m00'])
    cv2.circle(image, (x, y), 7, (255, 255, 255), -1)
#contours = list[contours]
#contours = np.asarray(contours)
    cv2.drawContours(image,contours.astype(int),-1,(0,0,225),6)
cv2.imshow('image',image)
cv2.waitKey(0)


''ss
cnt = contours[0]
area = cv2.contourArea(cnt)
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
plt.imshow(approx)
plt.show()
#cv2.imwrite('C:/Users/hp/OneDrive/Desktop/book1.jpg',approx)

#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray image',gray)
'''