import cv2
from flask import Flask, render_template, Response
from image_processing import set_img_marker, make_wrapping_img, make_filtering_img, set_roi_area
from lane_detection import find_lane, get_lane_slope, draw_lane_lines, get_direction_slope, add_img_weighted
from serial_arduino import make_serial_connection, write_signal, check_order

app = Flask(__name__)

# 차량 연결 시, 변경
flag = False

if flag:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("./video/ex3.mp4")

speed = 0
deg = 0
correction = -7

connection = None
order_msg = ""

current_test = 0


def make_test_image():
    global current_test

    while True:
        # TODO 프레임 값 수정 필요
        ret, img = cap.read()
        key = cv2.waitKey(30)

        if not ret:
            break

        (h, w) = (img.shape[0], img.shape[1])

        mark_img, src_position = set_img_marker(img)
        wrap_img, minv = make_wrapping_img(img, src_position)
        filter_img = make_filtering_img(wrap_img)
        roi_img = set_roi_area(filter_img)
        roi_in_img = set_roi_area(wrap_img)

        test_list = [mark_img, wrap_img, filter_img, roi_img, roi_in_img]
        result = test_list[current_test]

        # <<-- for Web streaming
        ret, buffer = cv2.imencode(".jpg", result)
        result = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + result + b'\r\n')
        # -->>


def make_image(image):
    (h, w) = (image.shape[0], image.shape[1])

    mark_img, src_position = set_img_marker(image)
    wrap_img, minv = make_wrapping_img(image, src_position)
    filter_img = make_filtering_img(wrap_img)
    roi_img = set_roi_area(filter_img)

    return minv, roi_img


def get_lane_information(original_image, roi_image, minv):
    global deg

    _DEG_ERROR_RANGE = 1
    _DIST_ERROR_RANGE = 3

    left, right = find_lane(roi_image)
    draw_info = get_lane_slope(roi_image, left, right)
    pts_mean, color_warp = draw_lane_lines(roi_image, minv, draw_info)
    deg, dist, color_warp = get_direction_slope(pts_mean, color_warp)
    result = add_img_weighted(original_image, color_warp, minv)

    flag = dist > _DIST_ERROR_RANGE
    deg = deg if flag else 0

    direction = "LEFT" if ((deg < _DEG_ERROR_RANGE * -1) and flag) \
        else ("RIGHT" if ((deg > _DEG_ERROR_RANGE) and flag)
              else "FRONT")

    left = int(original_image.shape[1] * 0.03)
    top = int(original_image.shape[0] * 0.1)

    cv2.putText(result, f"[{direction}]", (left, top),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"Deg : {deg}", (left, top + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"Dist : {dist}", (left, top + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return result, flag, deg


def gen_frames():
    global flag
    global speed
    global deg
    global correction
    global order_msg

    while True:
        # TODO 프레임 값 수정 필요
        ret, img = cap.read()
        key = cv2.waitKey(30)

        if not ret:
            break

        # MEMO 해상도에 따라 이미지 리사이징 필요
        # img = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        minv, roi_img = make_image(img)

        try:
            result, flag, deg = get_lane_information(img, roi_img, minv)
            order_msg = check_order(speed, deg + correction)

            left = int(result.shape[1] * 0.03)
            top = int(result.shape[0] * 0.7)

            # TODO 차량 연결 시, 활성화
            if flag:
                write_signal(connection, order_msg)
            else:
                write_signal("connection", order_msg)

        except:
            result = img

        cv2.putText(result, f"{order_msg}", (left, top),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, f"Throttles : {speed}", (left, top + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, f"Deg_corr : {correction}", (left, top + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # <<-- for Web streaming
        ret, buffer = cv2.imencode(".jpg", result)
        result = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + result + b'\r\n')
        # -->>


@app.route('/result_video')
def result_video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/test_video")
def test_video():
    return Response(make_test_image(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/test')
def test():
    return render_template("test.html")


@app.route('/start')
def start():
    global speed
    speed = 9
    return ("nothing")


@app.route('/stop')
def stop():
    global speed
    speed = 0
    return ("nothing")


@app.route('/speed_up')
def speed_up():
    global speed
    speed += 1
    speed = 13 if speed > 12 else speed
    return ("nothing")


@app.route('/speed_down')
def speed_down():
    global speed
    speed -= 1
    speed = 0 if speed < 0 else speed
    return ("nothing")


@app.route('/correction_plus')
def correction_plus():
    global correction
    correction += 1
    return ("nothing")


@app.route('/correction_minus')
def correction_minus():
    global correction
    correction -= 1
    return ("nothing")


@app.route('/mark_img')
def mark_img():
    global current_test
    current_test = 0
    return ("nothing")


@app.route('/wrap_img')
def wrap_img():
    global current_test
    current_test = 1
    return ("nothing")


@app.route('/roi_wrap_img')
def roi_wrap_img():
    global current_test
    current_test = 4
    return ("nothing")


@app.route('/filter_img')
def filter_img():
    global current_test
    current_test = 2
    return ("nothing")


@app.route('/roi_img')
def roi_img():
    global current_test
    current_test = 3
    return ("nothing")


if __name__ == "__main__":
    # TODO 차량 연결 시, 활성화
    # connection = make_serial_connection()
    app.run(host="192.168.50.41", port=8080)
    # main()
