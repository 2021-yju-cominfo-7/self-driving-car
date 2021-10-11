import numpy as np
import cv2

# direction = 0  # right
flag = False
speed = 0

while True:
    key = cv2.waitKey(30)

    if key == 32:
        flag = False if flag else True
        speed = 9 if flag else 0

    # 방향키 방향전환
    # elif key == 3:  # right
    #     direction = 0
    elif key == 1:  # down
        speed -= 1
    # elif key == 2:  # left
    #     direction = 2
    elif key == 0:  # up
        speed += 1

    speed = 0 if speed < 0 else speed
    flag = True if speed > 0 else False

    print(flag, speed)

    # 지우고, 그리기
    img = np.zeros((512, 512, 3), np.uint8) + 255  # 지우기
    cv2.imshow('controller', img)

cv2.destroyAllWindows()
