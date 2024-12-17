## Prompt:
Write a Python to solve Cifar classification problem:
1. Use "tensorflow.keras" to download Cifar dataset.
2. Use pre-trained VGG16 and VGG19 models.
3. Use figure to visualize training & validation accuracy for VGG16 and VGG19.

## Response:
Here's a Python code that solves the CIFAR classification problem using TensorFlow/Keras. This code:

1.  Downloads the CIFAR dataset.
2.  Uses pre-trained VGG16 and VGG19 models.
3.  Visualizes the training and validation accuracy for both models.

```python
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# 1. Download CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Function to create a model using pre-trained VGG16 or VGG19
def create_model(base_model):
    model = Sequential()
    
    # Add pre-trained base model
    model.add(base_model)
    
    # Add custom layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# 3. Load VGG16 and VGG19 pre-trained models (without the top layer)
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the layers of the pre-trained models to prevent retraining them
for layer in vgg16_base.layers:
    layer.trainable = False

for layer in vgg19_base.layers:
    layer.trainable = False

# 4. Create models
vgg16_model = create_model(vgg16_base)
vgg19_model = create_model(vgg19_base)

# 5. Train VGG16 model
history_vgg16 = vgg16_model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), batch_size=64)

# 6. Train VGG19 model
history_vgg19 = vgg19_model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), batch_size=64)

# 7. Visualize Training & Validation Accuracy
plt.figure(figsize=(12, 6))

# Plot VGG16 training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history_vgg16.history['accuracy'], label='VGG16 Training Accuracy')
plt.plot(history_vgg16.history['val_accuracy'], label='VGG16 Validation Accuracy')
plt.title('VGG16 - Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot VGG19 training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history_vgg19.history['accuracy'], label='VGG19 Training Accuracy')
plt.plot(history_vgg19.history['val_accuracy'], label='VGG19 Validation Accuracy')
plt.title('VGG19 - Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```

### Explanation:

1.  **Dataset**: The CIFAR-10 dataset is loaded using `cifar10.load_data()`. It contains 60,000 32x32 color images in 10 classes, with 6,000 images per class.
    
2.  **Pre-trained Models**: The pre-trained VGG16 and VGG19 models (without the top classification layer) are loaded using `VGG16()` and `VGG19()` from `tensorflow.keras.applications`. These models are pre-trained on the ImageNet dataset. We freeze the layers of these models so they do not get retrained during our fine-tuning process.
    
3.  **Model Creation**: The pre-trained models are added to a custom sequential model. After the VGG base, a `Flatten` layer is added, followed by a `Dense` layer with 512 neurons and ReLU activation, a `Dropout` layer to prevent overfitting, and a final `Dense` layer with 10 neurons for CIFAR-10 classification (using softmax activation).
    
4.  **Training**: Both models (VGG16 and VGG19) are trained on the CIFAR-10 dataset for 20 epochs, with a batch size of 64. We use the Adam optimizer and categorical cross-entropy loss function.
    
5.  **Visualization**: Training and validation accuracy are plotted for both VGG16 and VGG19 to visualize the model's performance over the epochs.
    

### Requirements:

Make sure you have the following libraries installed:

```bash
pip install tensorflow matplotlib
```

Let me know if you need further adjustments!
