import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap. read()
    key = cv2.waitKey(30)

    (h, w) = (img.shape[0], img.shape[1])

    if not ret:
        break

    center = int(w/2)
    top = int(h * 0.1)
    btm = int(h * 0.6)
    left = int(w * 0.36)
    right = int(w * 0.64)
    
    position = np.array([
        (w * 0.0, h * 0.6), (w * 0.36, h * 0.1), (w * 1.0, h * 0.6), (w * 0.64, h * 0.1)
    ])
    t_position = position.astype(int)

    cv2.circle(img, t_position[0], 10, (255,0,0), -1)
    cv2.circle(img, t_position[1], 10, (0,255,0), -1)
    cv2.circle(img, t_position[2], 10, (0,0,255), -1)
    cv2.circle(img, t_position[3], 10, (255, 255, 255), -1)

    cv2.imshow("1", img)
    source = np.float32(position)

    destination_position = [(w * 0.15, h * 0.95), (w * 0.15, h * 0.15), (w * 0.85, h * 0.95), (w * 0.85, h * 0.15)]
    destination = np.float32(destination_position)

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minverse = cv2.getPerspectiveTransform(destination, source)
    wrapped_img = cv2.warpPerspective(img, transform_matrix, (w, h))

    cv2.imshow("test", wrapped_img)

    if key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()