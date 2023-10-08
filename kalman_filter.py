from utils import *
import numpy as np
from tqdm import tqdm  

class KalmanFilter3D:
    def __init__(self, initial_state):
        self.state_estimate = np.array(initial_state)
        self.estimate_error = np.eye(3)  # Initial estimate error covariance matrix (3x3)
        self.measurement_noise = np.eye(3)  # Measurement noise covariance matrix (3x3)
        self.process_noise = np.eye(3) * 0.05  # Process noise covariance matrix (3x3)

    def predict(self):
        # predicter
        predicted_state = self.state_estimate
        predicted_estimate_error = self.estimate_error + self.process_noise
        return predicted_state, predicted_estimate_error

    def update(self, measurement):
        # updater
        predicted_state, predicted_estimate_error = self.predict()
        kalman_gain = np.dot(predicted_estimate_error, np.linalg.inv(predicted_estimate_error + self.measurement_noise))
        self.state_estimate = predicted_state + np.dot(kalman_gain, (measurement - predicted_state))
        self.estimate_error = np.dot((np.eye(3) - kalman_gain), predicted_estimate_error)

    def filter_measurements(self, measurements):
        # actual filtration method 
        filtered_states = []
        last_valid = []

        for measurement in tqdm(measurements, desc="Filtering Data"):
            if np.all(np.isnan(measurement)):
                try:
                    filtered_states.append(last_valid[len(last_valid) - 1])
                except IndexError:
                    filtered_states.append(np.zeros((3,), dtype=int))
            else:
                self.update(measurement)
                last_valid.append(self.state_estimate)
                filtered_states.append(self.state_estimate)

        return np.array(filtered_states)

if __name__ == "__main__":
    raise CodeNotWrittenError(message="There is no example here.")


