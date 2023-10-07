from utils import *
import numpy as np
import data_manager as dm
from tqdm import tqdm  
import matplotlib.pyplot as plt 


# class KalmanFilter:
#     def __init__(self, init_m, second_m):
#         pass
    
#     def filter_dataset(self):
#         pass

# class LowPassFilter: 
#     def __init__(self):
#         pass

# class MovingAvgFilter:
#     def __init__(self):
#         pass

def kalman_filter(initial_state, initial_estimate_error, measurement_noise, process_noise, measurements):
    # Initialize state variables
    state_estimate = initial_state
    estimate_error = initial_estimate_error
    
    filtered_states = []

    for measurement in tqdm(measurements, desc="Filtering Data"):
        # Prediction Step
        predicted_state = state_estimate
        predicted_estimate_error = estimate_error + process_noise

        # Update Step
        kalman_gain = np.dot(predicted_estimate_error, np.linalg.inv(predicted_estimate_error + measurement_noise))
        state_estimate = predicted_state + np.dot(kalman_gain, (measurement - predicted_state))
        estimate_error = np.dot((np.eye(len(initial_state)) - kalman_gain), predicted_estimate_error)

        filtered_states.append(state_estimate)

    return filtered_states

# Example usage
if __name__ == "__main__":
    # Simulated measurements (3D vectors)
    measurements = [dm.get_mag_field_vec(x) for x in tqdm(range(len(dm.data)), desc="Getting Measurements")]
    dates = [dm.get_datetime(x) for x in tqdm(range(len(dm.data)), desc="Getting Dates")]

    initial_state = np.array([measurements[12000][0], measurements[12000][1], measurements[12000][2]])  # Initial state estimate (3D vector)
    initial_estimate_error = np.eye(3)  # Initial estimate error covariance matrix (3x3)
    measurement_noise = np.eye(3)  # Measurement noise covariance matrix (3x3)
    process_noise = np.eye(3) * 0.05  # Process noise covariance matrix (3x3)

    filtered_states = kalman_filter(initial_state, initial_estimate_error, measurement_noise, process_noise, measurements)

    fig, ax1 = plt.subplots()

    # Plot the first data on the first Y-axis (left)
    ax1.plot(dates[12000:13000], measurements[12000:13000], color='tab:blue')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y1-axis', color='tab:blue')

    # Create a second set of Y-axes that shares the same X-axis
    ax2 = ax1.twinx()

    # Plot the second data on the second Y-axis (right)
    ax2.plot(dates[12000:13000], filtered_states[12000:13000], color='tab:red')
    ax2.set_ylabel('Y2-axis', color='tab:red')

    # Add a title
    plt.title('Graph')

    # Show the plot
    plt.show()
