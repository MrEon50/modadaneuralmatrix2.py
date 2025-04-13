# modadaneuralmatrix2.py

ModularAdaptiveNeuralMatrix
A lightweight, modular, and adaptive neural network framework implemented in pure Python without external dependencies.

The concept of a small modular adaptive neural network - about 40 MB (a cell with the possibility of scaling) not too big not too small able to handle a small mp4 file.

Overview
ModularAdaptiveNeuralMatrix is a self-contained neural network library designed for educational purposes and small to medium-sized machine learning projects. It provides a complete implementation of neural networks from scratch, including matrix operations, activation functions, loss functions, and optimization algorithms.

Key Features
Zero External Dependencies: Core functionality works without any external libraries
Matrix Operations: Custom implementation of essential matrix operations
Modular Architecture: Easy to extend and customize with new components
Adaptive Learning: Self-adjusting learning rates and network structures
Comprehensive Components:
Various activation functions (ReLU, Sigmoid, Tanh, Softmax)
Common loss functions (MSE, Binary/Categorical Cross-Entropy)
Multiple optimization algorithms (SGD, Momentum, Adam)
Regularization techniques (Dropout, Batch Normalization)
Self-adapting layers
Installation
Simply download the modadaneuralmatrix2.py file and import it into your project:

from modadaneuralmatrix2 import NeuralNetwork, Linear, Activation, ModelFactory

Copy


Basic Usage
Creating a Simple Neural Network
# Create a model for binary classification
model = NeuralNetwork()
model.add(Linear(input_size=4, output_size=8))
model.add(Activation("relu"))
model.add(Linear(8, 1))
model.add(Activation("sigmoid"))

# Compile the model
model.compile(
    loss="binary_cross_entropy",
    learning_rate=0.01,
    optimizer="adam"
)

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)

# Make predictions
predictions = model.predict(x_test)

Copy


Using Model Factory
# Create an MLP with predefined architecture
model = ModelFactory.create_mlp(
    input_size=10,
    hidden_sizes=[16, 8],
    output_size=3,
    dropout_rate=0.2,
    batch_norm=True
)

# Create an adaptive network
adaptive_model = ModelFactory.create_adaptive_network(
    input_size=10,
    hidden_sizes=[16, 8],
    output_size=3
)

Copy


Advanced Features
Data Processing
processor = DataProcessor()

# Normalize data
X_normalized, means, stds = processor.normalize(X)

# One-hot encode labels
y_encoded = processor.one_hot_encode(y, num_classes=3)

# Split data
X_train, X_test, y_train, y_test = processor.train_test_split(
    X_normalized, y_encoded, test_size=0.2
)

Copy


Model Saving and Loading
# Save model
model.save("my_model.pkl")

# Load model
new_model = NeuralNetwork()
new_model.load("my_model.pkl")

Copy


Transfer Learning
# Create a new model
target_model = NeuralNetwork()
# ... add layers ...

# Transfer weights from source model and freeze first 2 layers
target_model.transfer_learning(source_model, freeze_layers=2)

Copy


Example: XOR Problem
The library includes a built-in example solving the XOR problem:

from modadaneuralmatrix2 import example_xor

# Train and evaluate a model on the XOR problem
model = example_xor()
model.summary()

Copy


Optional Dependencies
While the core functionality works without external libraries, some advanced features benefit from:

NumPy: For more efficient data handling
Matplotlib: For visualization functions
scikit-learn: For additional data processing and example datasets
Limitations
Performance is not optimized for very large datasets or deep networks
No GPU acceleration
Limited to feed-forward neural networks
License
This project is available under the MIT License.

Acknowledgements
This library was created for educational purposes to demonstrate neural network principles from the ground up.
