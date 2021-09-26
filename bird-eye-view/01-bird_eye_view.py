import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("../img/playground/pg_center.jpeg")
(h, w) = (img.shape[0], img.shape[1])
# print(h, w)

size = 50

left = {"top": (1500, 1000), "bottom": (200, 1600)}
right = {"top": (2350, 1000), "bottom": (3150, 1600)}
color = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "black": (0, 0, 0)}

# 좌표점은 좌상->좌하->우상->우하
pts1 = np.float32([left["top"], left["bottom"], right["top"], right["bottom"]])

# 좌표의 이동점
pts2 = np.float32([[10, 10], [10, 1000], [1000, 10], [1000, 1000]])

# pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
cv2.circle(img, left["top"], size, color["red"], -1)
cv2.circle(img, left["bottom"], size, color["green"], -1)
cv2.circle(img, right["top"], size, color["blue"], -1)
cv2.circle(img, right["bottom"], size, color["black"], -1)

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img, M, (1100, 1100))

plt.subplot(121), plt.imshow(img), plt.title('image')
plt.subplot(122), plt.imshow(dst), plt.title('Perspective')
plt.show()
