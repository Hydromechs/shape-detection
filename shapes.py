import cv2
from shapedetector import ShapeDetector

image = cv2.imread('rectangles.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

ksize = 5
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize,ksize))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# ~ cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = cnts[1]
sd = ShapeDetector()

for c in cnts:
    M = cv2.moments(c)
    if M["m00"] != 0:  # prevent divide by zero
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        shape = sd.detect(c)

    c = c.astype("float")
    #c *= ratio
    c = c.astype("int")
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 0, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
