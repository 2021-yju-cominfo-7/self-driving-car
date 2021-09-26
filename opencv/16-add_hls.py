import cv2
import numpy as np
from image_processing import color_filter, find_edges
from util import get_line, get_fitline, draw_lines, draw_fit_line, weighted_img

cap = cv2.VideoCapture("../video/ex3.mp4")

winname = "image"
cv2.namedWindow(winname)
cv2.moveWindow(winname, 50, 500)

while True:
    ret, img = cap.read()

    if img is None:
        break

    try:
        temp = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        filtered = color_filter(img)
        edged = find_edges(filtered, 130, 255, 30, 350)

        L_lines, R_lines = get_line(edged)
        left_fit_line = get_fitline(img, L_lines)
        right_fit_line = get_fitline(img, R_lines)

        draw_lines(temp, L_lines, [0, 0, 255])
        draw_lines(temp, R_lines, [255, 0, 0])

        draw_fit_line(temp, left_fit_line, [0, 255, 0])
        draw_fit_line(temp, right_fit_line, [0, 255, 0])

        center_line = [
            int(np.mean([right_fit_line[2], left_fit_line[0]])),
            int(np.mean([right_fit_line[3], left_fit_line[1]])),
            int(np.mean([right_fit_line[0], left_fit_line[2]])),
            int(np.mean([right_fit_line[1], left_fit_line[3]]))
        ]
        print(left_fit_line, center_line, right_fit_line)
        draw_fit_line(temp, center_line, [0, 255, 0])
        result = weighted_img(temp, img)
    except:
        result = img

    cv2.imshow(winname, result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
