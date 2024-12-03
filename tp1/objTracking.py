from KalmanFilter import KalmanFilter
from Detector import detect
import cv2


if __name__ == "__main__":
    kf = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)
    trajectory = []
    cap = cv2.VideoCapture('randomball.avi')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        centers = detect(frame)
        for center in centers:
            kf.predict()
            predicted_x, predicted_y = map(int, kf.x_k[:2].flatten())
            cv2.rectangle(frame, (predicted_x - 10, predicted_y - 10), (predicted_x + 10, predicted_y + 10), (255, 0, 0), 2)  # draw the predicted object position

            kf.update(center)
            estimated_x, estimated_y = map(int, kf.x_k[:2].flatten())
            cv2.rectangle(frame, (estimated_x - 10, estimated_y - 10), (estimated_x + 10, estimated_y + 10), (0, 0, 255), 2)  # draw the estimated object position

            trajectory.append((estimated_x, estimated_y))
            for point in trajectory:
                cv2.circle(frame, point, 1, (0, 0, 0), -1)  # draw the trajectory

            cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)  # draw the detected circle

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # press q to quit sooner.
    cap.release()
    cv2.destroyAllWindows()
