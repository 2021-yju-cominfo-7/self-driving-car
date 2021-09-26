import cv2
import numpy as np


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):
    mask = np.zeros_like(img)
    color = color3 if len(img.shape) > 2 else color1

    cv2.fillPoly(mask, vertices, color)
    return cv2.bitwise_and(img, mask)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    return lines


def get_fitline(img, f_lines):
    lines = np.squeeze(f_lines)
    lines = lines.reshape(lines.shape[0] * 2, 2)

    rows, cols = img.shape[:2]
    output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]

    x1, y1 = int(((img.shape[0]-1)-y)/vy*vx + x), img.shape[0]-1
    x2, y2 = int(((img.shape[0]/2+100)-y)/vy*vx + x), int(img.shape[0]/2+100)

    return [x1, y1, x2, y2]


def draw_lines(img, lines, color=[0, 0, 255], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_fit_line(img, lines, color=[255, 0, 0], thickness=5):
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


def weighted_img(img, initial_img, α=1, β=1., λ=0.):
    # return cv2.addWeighted(initial_img, α, img, β, λ)
    return cv2.addWeighted(initial_img, 0.3, img, 1, 1)


video = cv2.VideoCapture("../video/ex3.mp4")
# nr_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

end_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("Video")
cv2.moveWindow("Video", 500, 500)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 500, 100)

cv2.namedWindow("Hough", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hough", 500, 100)

playSpeed = 5
playDelay = 100

# <<-- Create TrackBar
# cv2.createTrackbar("Frame", "Video", 0, nr_of_frames, getFrame)

cv2.createTrackbar("GaussianBlur", "Image", 0, 20, lambda x: x)
cv2.setTrackbarPos("GaussianBlur", "Image", 5)
cv2.createTrackbar("MedianBlur", "Image", 0, 20, lambda x: x)
cv2.setTrackbarPos("MedianBlur", "Image", 11)
cv2.createTrackbar("Thresh", "Image", 0, 200, lambda x: x)
cv2.setTrackbarPos("Thresh", "Image", 170)
cv2.createTrackbar("ThreshMax", "Image", 0, 255, lambda x: x)
cv2.setTrackbarPos("ThreshMax", "Image", 255)

cv2.createTrackbar("Threshold", "Hough", 0, 100, lambda x: x)
cv2.setTrackbarPos("Threshold", "Hough", 30)
cv2.createTrackbar("MinLength", "Hough", 0, 200, lambda x: x)
cv2.setTrackbarPos("MinLength", "Hough", 10)
cv2.createTrackbar("MaxGap", "Hough", 0, 200, lambda x: x)
cv2.setTrackbarPos("MaxGap", "Hough", 20)
# -->>

while video.isOpened():
    ret, image = video.read()

    if ret:
        # <<-- Get Value from TrackBar
        kernelSize = cv2.getTrackbarPos("GaussianBlur", "Image")
        kernelSize = kernelSize if kernelSize % 2 != 0 else kernelSize + 1
        medianValue = cv2.getTrackbarPos("MedianBlur", "Image")
        medianValue = medianValue if medianValue % 2 != 0 else medianValue + 1
        thresh = cv2.getTrackbarPos("Thresh", "Image")
        threshMax = cv2.getTrackbarPos("ThreshMax", "Image")

        threshold = cv2.getTrackbarPos("Threshold", "Hough")
        minLength = cv2.getTrackbarPos("MinLength", "Hough")
        maxGap = cv2.getTrackbarPos("MaxGap", "Hough")
        # -->>

        # <<-- Image Processing
        grayImg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurImg = cv2.GaussianBlur(grayImg, (kernelSize, kernelSize), 0)
        medianImg = cv2.medianBlur(blurImg, medianValue)
        thrRet, thrImg = cv2.threshold(medianImg, thresh, threshMax, cv2.THRESH_BINARY)
        # TODO 트랙바 추가하기
        cannyImg = cv2.Canny(thrImg, 30, 350)
        # -->>

        # <<-- Set ROI
        # vertices = np.array(
        #     [[(leftX, bottomY), (leftX + leftDegree, topY), (rightX - rightDegree, topY), (rightX, bottomY)]],
        #     dtype=np.int32)

        vertices = np.array(
            [[(50, height / 2 + 100), (width / 2 - 200, 70), (width / 2 + 200, 70), (width - 50, height / 2 + 100)]],
            dtype=np.int32)
        roi_test = region_of_interest(image, vertices)
        roi_img = region_of_interest(cannyImg, vertices, (0, 0, 255))
        # -->>

        try:
            # <<-- Probabilistic Hough Transform
            line_arr = hough_lines(roi_img, 1, 1 * np.pi / 180, threshold, minLength, maxGap)
            line_arr = np.squeeze(line_arr)
            # -->>

            # 기울기 구하기
            slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

            # 수평 기울기 제한
            line_arr = line_arr[np.abs(slope_degree) < 160]
            slope_degree = slope_degree[np.abs(slope_degree) < 160]

            # 수직 기울기 제한
            line_arr = line_arr[np.abs(slope_degree) > 95]
            slope_degree = slope_degree[np.abs(slope_degree) > 95]

            # 필터링된 직선 버리기
            L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
            temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            L_lines, R_lines = L_lines[:, None], R_lines[:, None]

            left_fit_line = get_fitline(image, L_lines)
            right_fit_line = get_fitline(image, R_lines)

            # TODO 대표선이 삐뚤어진 경우를 계산!!!
            print(left_fit_line)

            draw_lines(temp, L_lines)
            draw_lines(temp, R_lines)

            draw_fit_line(temp, left_fit_line)
            draw_fit_line(temp, right_fit_line)

            result = weighted_img(temp, image)
        except:
            result = roi_test

        cv2.imshow("Video1", roi_test)
        cv2.imshow("Video2", result)
        frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))

        if frame == end_frame:
            video.set(cv2.CAP_PROP_POS_FRAMES, 1)

        if frame % playDelay == 0 or frame == 10:
            print(f"------------------------------ Frame : {frame} ------------------------------")
            print(f"KernelSize : {kernelSize} / Median : {medianValue} / Thresh : {thresh} / ThreshMax : {threshMax}")
            # print(f"ROI / X : ({leftX, rightX}), Y : ({topY, bottomY}), D : ({leftDegree, rightDegree})")
            print(f"HoughLines / {threshold}, MinLength : {minLength}, MaxGap : {maxGap}")
            # cv2.setTrackbarPos("Frame", "Video", int(video.get(cv2.CAP_PROP_POS_FRAMES)))
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
