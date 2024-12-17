## Prompt:
Write a Python to solve MNIST classification problem:
1. Use "tensorflow.keras.datasets" to download MNIST dataset.
2. Use "tensorflow.keras" to train Dense NN and CNN model.
3. Use plot to visualize training & validation accuracy for Dense NN and CNN.

## Response:
Here's a Python script to solve the MNIST classification problem using both a Dense Neural Network (DNN) and a Convolutional Neural Network (CNN). It utilizes TensorFlow/Keras for model creation and training, as well as `matplotlib` to visualize the training and validation accuracy.

```python
# Importing necessary libraries
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing the data (normalize the data)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshaping the data for CNN (adding a channel dimension)
x_train_cnn = x_train.reshape((-1, 28, 28, 1))
x_test_cnn = x_test.reshape((-1, 28, 28, 1))

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 2. Define Dense Neural Network (DNN) model
def create_dense_nn():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images
        layers.Dense(128, activation='relu'),  # Fully connected layer with ReLU activation
        layers.Dropout(0.2),  # Dropout for regularization
        layers.Dense(10, activation='softmax')  # Output layer with 10 classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. Define Convolutional Neural Network (CNN) model
def create_cnn():
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 4. Train Dense NN model
dense_nn_model = create_dense_nn()
history_dense_nn = dense_nn_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# 5. Train CNN model
cnn_model = create_cnn()
history_cnn = cnn_model.fit(x_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(x_test_cnn, y_test))

# 6. Plotting the training and validation accuracy for both models
def plot_accuracy(history, model_name):
    plt.plot(history.history['accuracy'], label=f'{model_name} Training Accuracy')
    plt.plot(history.history['val_accuracy'], label=f'{model_name} Validation Accuracy')
    plt.title(f'{model_name} Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot accuracy for Dense NN model
plot_accuracy(history_dense_nn, "Dense NN")

# Plot accuracy for CNN model
plot_accuracy(history_cnn, "CNN")
```

### Explanation of the code:

1.  **Loading and Preprocessing Data**:
    
    *   The MNIST dataset is loaded using `tensorflow.keras.datasets.mnist`.
    *   The images are normalized to a range between 0 and 1 by dividing by 255.0.
    *   The images are reshaped for CNN models, which require the shape `(batch_size, height, width, channels)`.
    *   The labels are converted to one-hot encoding using `tf.keras.utils.to_categorical`.
2.  **Creating Models**:
    
    *   **Dense Neural Network (DNN)**:
        *   A simple feedforward neural network with one hidden layer containing 128 units and a dropout layer for regularization.
        *   The output layer has 10 units for classification (one for each MNIST digit).
    *   **Convolutional Neural Network (CNN)**:
        *   Three convolutional layers with increasing filter sizes (32, 64, 128 filters).
        *   Max-pooling layers follow each convolutional layer to reduce spatial dimensions.
        *   Flattening and dense layers are used after the convolutional blocks, with a dropout for regularization.
3.  **Training Models**:
    
    *   Both models are compiled with the Adam optimizer and categorical cross-entropy loss function.
    *   Training is done for 10 epochs with a batch size of 32. Validation accuracy is computed during training using the test set.
4.  **Plotting Accuracy**:
    
    *   The training and validation accuracy for both models (Dense NN and CNN) are plotted using `matplotlib` to compare their performance.

### How to Run:

1.  Install the necessary libraries if you haven't already:
    
    ```bash
    pip install tensorflow matplotlib
    ```
    
2.  Run the script in any Python environment with access to TensorFlow (e.g., Jupyter Notebook or a Python script).

This code will visualize the accuracy of both models over 10 epochs and allow you to compare the performance of the Dense NN and CNN on the MNIST dataset.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)