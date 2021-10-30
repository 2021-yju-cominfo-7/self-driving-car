import cv2
import numpy as np


def img_contrast_ver1(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final


def img_contrast_ver2(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img2 = clahe.apply(img)

    final = np.hstack((img, img2))

    return final


cap = cv2.VideoCapture(3)

while True:
    ret, img = cap.read()
    key = cv2.waitKey(30)

    if not ret:
        break

    img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    result = img_contrast_ver2(img)

    cv2.imshow("img", img)
    cv2.imshow("result", result)

    if key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
