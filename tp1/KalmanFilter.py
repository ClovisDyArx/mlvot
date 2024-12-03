import cv2
import numpy as np


class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.dt = dt
        self.std_acc = std_acc
        self.x_std_meas = x_std_meas
        self.y_std_meas = y_std_meas

        self.u = np.array([[u_x], [u_y]])
        self.x_k = np.array([[0], [0], [0], [0]])

        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.B = np.array([[0.5 * self.dt ** 2, 0],
                           [0, 0.5 * self.dt ** 2],
                           [self.dt, 0],
                           [0, self.dt]])

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        self.Q = (np.array([[self.dt ** 4 / 4, 0, self.dt ** 3 / 2, 0],
                           [0, self.dt ** 4 / 4, 0, self.dt ** 3 / 2],
                           [self.dt ** 3 / 2, 0, self.dt ** 2, 0],
                           [0, self.dt ** 3 / 2, 0, self.dt ** 2]])
                  * self.std_acc ** 2)

        self.R = np.array([[self.x_std_meas ** 2, 0],
                           [0, self.y_std_meas ** 2]])

        self.P_k = np.eye(self.A.shape[1])

    def predict(self):
        self.x_k = np.dot(self.A, self.x_k) + np.dot(self.B, self.u)  # update time state
        self.P_k = np.dot(np.dot(self.A, self.P_k), self.A.T) + self.Q  # calculate error covariance

    def update(self, z_k):
        Sk = np.dot(np.dot(self.H, self.P_k), self.H.T) + self.R
        Kk = np.dot(np.dot(self.P_k, self.H.T), np.linalg.inv(Sk))

        self.x_k = self.x_k + np.dot(Kk, z_k - np.dot(self.H, self.x_k))
        self.P_k = np.dot((np.eye(self.P_k.shape[0]) - np.dot(Kk, self.H)), self.P_k)
