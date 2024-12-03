from KalmanFilter import KalmanFilter
from Detector import detect
import cv2


if __name__ == "__main__":
    kf = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)

    cap = cv2.VideoCapture('randomball.avi')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        centers = detect(frame)
        for center in centers:
            kf.predict()
            kf.update(center)
            x, y = map(int, kf.x_k[:2].flatten())
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
