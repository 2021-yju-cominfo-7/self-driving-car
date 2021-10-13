from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(1)
speed = 0


def gen_frames():
    global speed
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        print(speed)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template("index.html")


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


if (__name__ == '__main__'):
    app.run(host="192.168.50.41", port=8080)
