from utils import *
import numpy as np
from tqdm import tqdm  
import matplotlib.pyplot as plt
import pandas as pd
import data_manager as dm


class KalmanFilter3D:
    def __init__(self, initial_state):
        # Initialize state variables as numpy arrays
        self.state_estimate = np.array(initial_state)
        self.estimate_error = np.eye(3)  # Initial estimate error covariance matrix (3x3)
        self.measurement_noise = np.eye(3)  # Measurement noise covariance matrix (3x3)
        self.process_noise = np.eye(3) * 0.05  # Process noise covariance matrix (3x3)

    def predict(self):
        # Prediction Step
        predicted_state = self.state_estimate
        predicted_estimate_error = self.estimate_error + self.process_noise
        return predicted_state, predicted_estimate_error

    def update(self, measurement):
        # Update Step
        predicted_state, predicted_estimate_error = self.predict()
        kalman_gain = np.dot(predicted_estimate_error, np.linalg.inv(predicted_estimate_error + self.measurement_noise))
        self.state_estimate = predicted_state + np.dot(kalman_gain, (measurement - predicted_state))
        self.estimate_error = np.dot((np.eye(3) - kalman_gain), predicted_estimate_error)

    def filter_measurements(self, measurements):
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
    pass

    # Example Usage (doesn't work coz of circular import)
    data = [np.array(dm.get_mag_field_vec(x)) for x in tqdm(range(len(dm.data)), desc="Getting Measurements")]
    filt = KalmanFilter3D([data[0][0], data[0][1], data[0][2]])
    filtered = filt.filter_measurements(data)

    fig, ax1 = plt.subplots()
    x = range(500)

    # # Plot the first data on the first Y-axis (left)
    ax1.plot(x, data[0:500], color='tab:blue')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y1-axis', color='tab:blue')

    # # Create a second set of Y-axes that shares the same X-axis
    ax2 = ax1.twinx()

    # # Plot the second data on the second Y-axis (right)
    ax2.plot(x, filtered[0:500], color='tab:red')
    ax2.set_ylabel('Y2-axis', color='tab:red')

    # Add a title
    plt.title('Graph')

    # Show the plot
    plt.show()


