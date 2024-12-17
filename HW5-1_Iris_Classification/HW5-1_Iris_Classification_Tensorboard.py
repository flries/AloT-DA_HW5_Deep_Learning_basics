# Import necessary libraries
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
import os

# Step 1: Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Labels (species: 0 - setosa, 1 - versicolor, 2 - virginica)

# Step 2: Preprocess the data
# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode the target labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the model using TensorFlow Keras
model = models.Sequential()

# Add an input layer with 4 features (as there are 4 features in the Iris dataset)
model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))

# Add hidden layers with ReLU activation
model.add(layers.Dense(8, activation='relu'))  # First hidden layer with 8 neurons
model.add(layers.Dense(8, activation='relu'))  # Second hidden layer with 8 neurons

# Add the output layer (3 neurons for the 3 classes)
model.add(layers.Dense(3, activation='softmax'))

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # As labels are integers (not one-hot encoded)
              metrics=['accuracy'])

# Step 5: Set up TensorBoard for visualization
log_dir = "logs/iris_classification"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Step 6: Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), 
                    callbacks=[tensorboard_callback])

# Step 7: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# To visualize the training process in TensorBoard, run this command in a terminal:
# tensorboard --logdir=logs/iris_classification
