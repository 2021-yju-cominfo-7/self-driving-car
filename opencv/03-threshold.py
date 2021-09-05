import cv2

src = cv2.imread("../img/line/line-1.png", cv2.IMREAD_COLOR)
dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, thr1 = cv2.threshold(dst, 170, 255, cv2.THRESH_BINARY)
ret, thr2 = cv2.threshold(dst, 170, 255, cv2.THRESH_BINARY_INV)
ret, thr3 = cv2.threshold(dst, 170, 255, cv2.THRESH_TRUNC)
ret, thr4 = cv2.threshold(dst, 170, 255, cv2.THRESH_TOZERO)
ret, thr5 = cv2.threshold(dst, 170, 255, cv2.THRESH_TOZERO_INV)

cv2.imshow("original", src)
cv2.imshow("dst", dst)
cv2.imshow("BINARY", thr1)
cv2.imshow("BINARY_INV", thr2)
cv2.imshow("TRUNC", thr3)
cv2.imshow("TOZERO", thr4)
cv2.imshow("TOZERO_INV", thr5)

cv2.waitKey()
cv2.destroyAllWindows()
