# Leaf Disease Detection Using Deep Learning

## Abstract

The proposed decision-making system utilizes image content characterization and a supervised classifier type of neural network. Image processing techniques for this kind of decision analysis involve preprocessing, feature extraction, and classification stages. During preprocessing, resize, color, and texture features are extracted from an input for training. The system will be used to classify the test images automatically to decide leaf characteristics.

## Scope of the Project

- **Data Analysis**
- **Data Preprocessing**
- **Training the Model**
- **Testing the Model**

## Working Model

In this model, the image is fed into the convolutional neural network (CNN) with a size of 128x128 and three color channels. The first convolutional layer (Conv2D) applies 32 filters to find features and create a feature map. After that, the ReLU activation function is applied to remove non-linearity. Batch normalization is then applied to normalize the weights of neurons. The feature map is then fed into a max-pooling layer, which extracts the most relevant features. This process is repeated with additional Conv2D and max-pooling layers. The output of these layers is then fed into a fully connected layer (linear layer). Finally, the output layer predicts 25 categories.

## Installation

Ensure you have Python installed (>=3.6). Install the required dependencies:

```sh
pip install numpy tensorflow matplotlib
```

## Usage

1. **Prepare Data**: Ensure you have the necessary dataset for training and testing.
2. **Load and Preprocess Data**: Load the data, resize, and normalize it.
3. **Build and Compile the Model**: Create a CNN model with convolutional, pooling, and dense layers. Compile the model with a loss function, optimizer, and metrics.
4. **Train the Model**: Train the model on the training data for a specified number of epochs.
5. **Evaluate the Model**: Evaluate the model on the testing data to check its performance.
6. **Make Predictions**: Use the trained model to make predictions on new data samples.

### Example Code

Here is an example of the complete code:

```python
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
import matplotlib.pyplot as plt
import random

# Load Data
x_train = np.loadtxt('input.csv', delimiter=',')
y_train = np.loadtxt('labels.csv', delimiter=',')
x_test = np.loadtxt('input_test.csv', delimiter=',')
y_test = np.loadtxt('labels_test.csv', delimiter=',')

# Reshape and Normalize Data
x_train = x_train.reshape(len(x_train), 128, 128, 3)
y_train = y_train.reshape(len(y_train), 1)
x_test = x_test.reshape(len(x_test), 128, 128, 3)
y_test = y_test.reshape(len(y_test), 1)
x_train = x_train / 255
x_test = x_test / 255

# Print Data Shapes
print("shape of x_train: ", x_train.shape)
print("shape of y_train: ", y_train.shape)
print("shape of x_test: ", x_test.shape)
print("shape of y_test: ", y_test.shape)

# Build and Compile Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(25, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
model.fit(x_train, y_train, epochs=5, batch_size=64)

# Evaluate Model
model.evaluate(x_test, y_test)

# Make Predictions
idx = random.randint(0, len(y_test))
plt.imshow(x_test[idx, :])
plt.show()

y_pred = model.predict(x_test[idx, :].reshape(1, 128, 128, 3))
print(y_pred)

pred_class = np.argmax(y_pred)
print(f"Our model says it is a class: {pred_class}")
```

## Future Enhancements

- Improve model accuracy with more complex architectures.
- Add support for more leaf categories.
- Implement data augmentation for better generalization.
- Integrate visualization tools for model performance.

## Author

Developed by Pavan Kalyan Manda

Website Developer | IoT & Embedded Systems Enthusiast | AI/ML Developer
