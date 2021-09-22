import cv2


def getFrame(frame_nr):
    global video
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)


def setSpeed(val):
    global playSpeed
    playSpeed = max(val, 1)


video = cv2.VideoCapture("../video/ex3.mp4")
nr_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
cv2.namedWindow("Video")

playSpeed = 5
playDelay = 50

cv2.createTrackbar("Frame", "Video", 0, nr_of_frames, getFrame)
cv2.createTrackbar("Speed", "Video", playSpeed, 100, setSpeed)

while video.isOpened():
    ret, frame = video.read()
    if ret:
        cv2.imshow("Video", frame)
        if video.get(cv2.CAP_PROP_POS_FRAMES) % playDelay == 0:
            cv2.setTrackbarPos("Frame", "Video", int(video.get(cv2.CAP_PROP_POS_FRAMES)))
    else:
        break

    key = cv2.waitKey(playSpeed)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
