import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np 
import data_manager as dm 
import matplotlib.pyplot as plt

# get training dataset --------
# dataset = dm.get_dataset(start_year=2017)[0] 
# X = np.array([arr[1][0] for arr in dataset]) # average filtered data (get past reshape issues)
# #print(X)
#Y = np.array([arr[0] for arr in dataset])  # kp values

# The TensorFlow Model: (Uncomment to process this) 
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# #print(X_train.shape)

# # set up tensorflow model --------
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(3, 1)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu', input_shape=(1,)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1)  # Output layer for regression task
# ])

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
# model.fit(X_train, Y_train, epochs=250, batch_size=32, validation_split=0.2)

# for testing purposes 
#loss, mae = model.evaluate(X_test, Y_test)
#print("Mean Absolute Error:", mae)

# model.save('data/models/predictor') # UNCOMMENT ALL ABOVE CODE TO GET THE MODEL 

#print(len(predictions))
# visualize (testing)
# plt.scatter(np.arange(len(predictions)), predictions, color='blue')
# plt.xlabel('X Values')
# plt.ylabel('Y Values')
# plt.legend()
# plt.show()

def get_neural_Kp_data(dX, YEAR):
    model = tf.keras.models.load_model('data//models//predictor')
    new_X = dX
    predictions = model.predict(new_X)
    predictions = [np.round(prediction) for prediction in predictions]
    with open(f'data/preds/{YEAR}_predictions.csv', "w", newline="\n") as f:
        for prediction in predictions:
            f.write(f"{int(prediction[0])},")

if __name__ == "__main__":
    data2 = dm.get_dataset(start_year=2023)[0]
    x = np.array([arr[0][0] for arr in data2])
    get_neural_Kp_data(x, 2023)
