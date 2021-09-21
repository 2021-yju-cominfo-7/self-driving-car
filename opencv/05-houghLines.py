import cv2
import numpy as np

src = cv2.imread("../img/line/line-1.png", cv2.IMREAD_COLOR)
dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, thr = cv2.threshold(dst, 170, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(thr, 30, 70)

# 직선 성분 검출
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 160, minLineLength=50, maxLineGap=50)

# 라인 정보를 받았으면
if lines is not None:
    for i in range(lines.shape[0]):
        pt1 = (lines[i][0][0], lines[i][0][1])  # 시작점 좌표 x,y
        pt2 = (lines[i][0][2], lines[i][0][3])  # 끝점 좌표, 가운데는 무조건 0
        cv2.line(src, pt1, pt2, (0, 0, 255), 5, cv2.LINE_AA)

cv2.imshow("lines", src)

cv2.waitKey()
cv2.destroyAllWindows()
