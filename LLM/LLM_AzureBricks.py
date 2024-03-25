# Step 1: Import necessary libraries
import tensorflow as tf  # Import TensorFlow library
import torch  # Import PyTorch library
import torchvision  # Import torchvision library for datasets and transformations
import torchvision.transforms as transforms  # Import transforms module for data preprocessing
import torch.nn as nn  # Import neural network module from PyTorch
import torch.optim as optim  # Import optimization module from PyTorch

# Step 2: Define a neural network architecture using TensorFlow
def create_tf_model():
    # Define a sequential model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the input images
        tf.keras.layers.Dense(128, activation='relu'),  # Add a dense layer with ReLU activation
        tf.keras.layers.Dropout(0.2),  # Add dropout layer to prevent overfitting
        tf.keras.layers.Dense(10)  # Add output layer
    ])
    return model

# Step 3: Define a neural network architecture using PyTorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Define first fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Define second fully connected layer

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten input tensor
        x = nn.functional.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = self.fc2(x)  # Pass through the second layer
        return x

# Step 4: Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # Define data transformations
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # Load training set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)  # Create data loader

# Step 5: Initialize models and define loss functions and optimizers
tf_model = create_tf_model()  # Initialize TensorFlow model
torch_model = Net()  # Initialize PyTorch model
loss_fn_tf = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # Define loss function for TensorFlow
criterion_torch = nn.CrossEntropyLoss()  # Define loss function for PyTorch
optimizer_tf = tf.keras.optimizers.Adam()  # Define optimizer for TensorFlow
optimizer_torch = optim.Adam(torch_model.parameters())  # Define optimizer for PyTorch

# Step 6: Define training steps for TensorFlow model
@tf.function
def train_tf_step(images, labels):
    with tf.GradientTape() as tape:
        logits = tf_model(images)
        loss = loss_fn_tf(labels, logits)
    gradients = tape.gradient(loss, tf_model.trainable_variables)
    optimizer_tf.apply_gradients(zip(gradients, tf_model.trainable_variables))
    return loss

# Step 7: Define training steps for PyTorch model
def train_torch_step(inputs, labels):
    optimizer_torch.zero_grad()
    outputs = torch_model(inputs)
    loss = criterion_torch(outputs, labels)
    loss.backward()
    optimizer_torch.step()
    return loss.item()

# Step 8: Main training loop
epochs = 5  # Define number of epochs
for epoch in range(epochs):
    running_loss_tf = 0.0  # Initialize running loss for TensorFlow model
    running_loss_torch = 0.0  # Initialize running loss for PyTorch model
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data  # Get inputs and labels
        # Train TensorFlow model
        loss_tf = train_tf_step(inputs, labels)
        running_loss_tf += loss_tf
        # Train PyTorch model
        loss_torch = train_torch_step(inputs, labels)
        running_loss_torch += loss_torch

    # Print average loss for each epoch
    print(f"Epoch {epoch + 1}, TensorFlow Loss: {running_loss_tf / len(trainloader)}, PyTorch Loss: {running_loss_torch / len(trainloader)}")

print('Finished Training')  # Print message after training completes
