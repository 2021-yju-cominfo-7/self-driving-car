import cv2

src = cv2.imread("../../img/line/line-1.png", cv2.IMREAD_COLOR)
dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, thr = cv2.threshold(dst, 170, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(thr, 30, 70)  # 하단 임계값과 상단 임계값은 실험적으로 결정하기
cv2.imshow("canny", edges)

cv2.waitKey()
cv2.destroyAllWindows()
