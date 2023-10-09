import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np 
import data_manager as dm 
import matplotlib.pyplot as plt

# get data
dataset = dm.get_dataset(start_year=2017)[0]
X = np.array([arr[1][0] for arr in dataset]) # average filtered data (get past reshape issues)
print(X)
Y = np.array([arr[0] for arr in dataset])  # kp values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape)

# set up tensorflow model 
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression task
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train, Y_train, epochs=250, batch_size=32, validation_split=0.2)

loss, mae = model.evaluate(X_test, Y_test)
#print("Mean Absolute Error:", mae)

data2 = dm.get_dataset(start_year=2016)[0]
new_X = np.array([arr[1][0] for arr in data2])
predictions = model.predict(new_X)
predictions = [np.round(prediction) for prediction in predictions]

plt.scatter(np.arange(len(Y)), Y, label='Original Data', color='blue')
plt.scatter(np.arange(len(predictions)), predictions, label='Predictions', color='red')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.legend()
plt.show()