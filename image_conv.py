import cv2

img = cv2.imread('full2.png', 0)
edges = cv2.Canny(img, 100, 200)

cv2.imwrite('full2canny.jpg', edges)