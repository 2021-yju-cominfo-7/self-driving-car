import cv2
import numpy as np


def draw_lines(img, lines, color=[0, 0, 255], thickness=5):  # 선 그리기
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    return lines


def roi(image):
    x = int(image.shape[1])
    y = int(image.shape[0])

    arr = [
        [int(0.1 * x), int(y)], [int(0.1 * x), int(0.1 * y)], [int(0.4 * x), int(0.1 * y)], [int(0.4 * x), int(y)],
        [int(0.7 * x), int(y)], [int(0.7 * x), int(0.1 * y)], [int(0.9 * x), int(0.1 * y)], [int(0.9 * x), int(y)],
        [int(0.1 * x), int(y)]
    ]
    # 한 붓 그리기
    _shape = np.array(arr)

    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def color_filter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    lower = np.array([20, 150, 20])
    upper = np.array([255, 255, 255])

    yellow_lower = np.array([0, 85, 81])
    yellow_upper = np.array([190, 255, 255])

    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(image, image, mask=mask)

    return masked


# Bird Eye View Wrapping
def wrapping(image):
    (h, w) = (image.shape[0], image.shape[1])

    # 좌상, 우상, 좌하, 우하
    arr = {
        "source": [[w // 2 - 60, h * 0.53], [w // 2 + 160, h * 0.53], [w * 0.3 - 200, h], [w - 100, h]],
        "destination": [[-200, 0], [w, 0], [200, h], [w - 350, h]]
    }
    source = np.float32(arr["source"])
    destination = np.float32(arr["destination"])

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minv = cv2.getPerspectiveTransform(destination, source)
    _image = cv2.warpPerspective(image, transform_matrix, (w, h))

    return _image, minv, arr


cap = cv2.VideoCapture("./블랙박스 영상.mp4")
size = 10
color = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "black": (0, 0, 0)}

while True:
    ret, img = cap.read()

    if not ret:
        break

    (h, w) = (img.shape[0], img.shape[1])

    _img, minv, arr = wrapping(img)
    result = list(map(lambda x: (int(x[0]), int(x[1])), arr["source"]))

    cv2.circle(img, result[0], size, color["red"], -1)
    cv2.circle(img, result[1], size, color["green"], -1)
    cv2.circle(img, result[2], size, color["blue"], -1)
    cv2.circle(img, result[3], size, color["black"], -1)

    _img = color_filter(_img)
    _img = roi(_img)

    _gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(_gray, 160, 255, cv2.THRESH_BINARY)
    line_arr = hough_lines(thresh, 1, 1 * np.pi / 180, 160, 50, 50)
    line_arr = np.squeeze(line_arr)

    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
    temp = np.zeros((_img.shape[0], _img.shape[1], 3), dtype=np.uint8)
    L_lines, R_lines = L_lines[:, None], R_lines[:, None]

    draw_lines(temp, L_lines)
    draw_lines(temp, R_lines)

    cv2.imshow("Bird Eye View", temp)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
