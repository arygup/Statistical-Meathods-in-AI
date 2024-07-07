# Comparison: MLP vs. CNN on Permuted and MultiMNIST Datasets

We will discuss the performance of Multi-Layer Perceptron (MLP) models compared to Convolutional Neural Network (CNN) models on two distinct datasets: the Permuted MNIST and MultiMNIST datasets. We will also address the increase in training time and its impact on hyperparameter tuning.


## Permuted MNIST Dataset

### MLP
- **MLP** models have shown superior performance on the **Permuted MNIST** dataset compared to CNN models. This could be attributed to the nature of the dataset, where pixels are randomly shuffled, making it challenging for CNNs to capture spatial dependencies.
- MLPs have fully connected layers, allowing them to model complex relationships between shuffled pixels more effectively.
- The MLP architecture used in this case comprised an input layer, a hidden layer, and an output layer. The hidden layer helps the model learn relevant features.

### CNN
- **CNN** models have more specialized layers, such as convolutional and pooling layers, which are designed for spatial feature extraction in images.
- While CNNs are powerful for regular image classification tasks, they struggle to capture the underlying structure when pixel positions are permuted.
- The model's convolutional and pooling operations might not be as effective in finding meaningful patterns in the shuffled data.

### Impact of Training Time
- The primary difference is the **time taken for training**. CNNs generally require more time to train compared to MLPs. 
- On the Permuted MNIST dataset, training CNNs can be time-consuming. Due to the need for complex weight sharing and feature extraction, each run may take significantly longer.
- This extended training time makes **hyperparameter tuning** more challenging. Optimizing hyperparameters often requires trying different configurations, and in the case of CNNs, it may not be feasible to experiment with numerous hyperparameter combinations within a reasonable timeframe.

## MultiMNIST Dataset

### MLP
- Similar to the Permuted MNIST dataset, MLPs also tend to outperform CNNs on the **MultiMNIST** dataset on the training set, but get **out performed in the test dataset.!**
- MultiMNIST consists of superimposed digits, making it more complex. MLPs can effectively capture the features required for classifying these superimposed digits.

### CNN
- CNNs can perform well on the MultiMNIST dataset, but it might require more complex architectures and deeper layers to handle the additional complexity introduced by superimposed digits.

### Impact of Training Time
- The increase in training time for CNNs is more pronounced in the case of the MultiMNIST dataset.
- Superimposed digits add an extra layer of complexity, and CNNs need to learn more intricate features. This results in even longer training times.
- Consequently, hyperparameter tuning becomes a time-consuming process, as experimenting with various combinations of hyperparameters takes considerably longer.

In summary, while CNNs are powerful models for image classification, their training time and complex architecture make them less suitable for datasets where spatial information is not critical. MLPs are faster to train and can provide competitive performance on such datasets. However, it's essential to consider the dataset's characteristics and the trade-off between training time and performance when choosing the appropriate model architecture.

## Manual Hyperparameter Tuning Process

Manually finding optimal hyperparameters by running the same code multiple times in an iterative process. The goal was to find decent hyperparameters that provided good results without extensive computational resources. 

#### Experimentation Workflow

1. **Initial Model Configuration**: I began with an initial model configuration for both MLP and CNN. These configurations included settings like learning rate, batch size, number of hidden layers, number of neurons in hidden layers, and number of epochs.

2. **Iterative Testing**: I ran the training code using these initial hyperparameters. 

3. **Analysis and Evaluation**: After training, I evaluated the model's performance on validation and test datasets. I looked metrics like loss and accuracy to assess how well the model was learning.

4. **Adjustment**: Based on the initial results, I adjusted one or more hyperparameters to explore different settings. This could include increasing or decreasing learning rates, changing batch sizes, or modifying the model architecture.

5. **Repeat Iteration**: I repeated steps 2-4 multiple times, each time with a slightly different configuration. This iterative process allowed me to learn from the previous runs and make more informed decisions about which hyperparameters to adjust.

6. **Balance Between Performance and Time**: It's important to mention that I didn't aim for the absolute best hyperparameters. Instead, I sought to find a balance between decent performance and reasonable training times. The goal was to identify hyperparameters that would provide satisfactory results without requiring days or weeks of training.



# MultiMNIST Dataset Preprocessing

## Overview
 The dataset consists of images of handwritten digits, but it has been modified to remove the digits 0 to 9 to ensure the model cannot train on the same digit twice.

### 1. Deleting Digits
To remove the digits 0 to 9 from the dataset, a specific data preprocessing step is taken. This is done to avoid the model training on identical digits. 

### 2. Loading the Dataset
The dataset is loaded from the 'double_mnist' directory, with separate subsets for training, testing, and validation. These subsets are loaded as ImageFolder datasets.

### 3. Extracting Labels from Image Names
This section defines a function to extract labels (Y values) from the image filenames. The labels are extracted based on the structure of the image filenames, which include two digits. The first digit represents one part of the label, and the second digit represents the other part of the label. The extracted labels are organized into y_train, y_val, and y_test for the respective subsets.

### 4. Plotting Images with Labels
This function allows visualization of random image samples from a dataset with their corresponding labels. Five random samples are chosen from the dataset, and the images and labels are displayed in a row for each sample.

### 5. Displaying Sample Images
Three separate calls to the function 'plot_images_with_filenames' are made to visualize sample images and their associated labels for the training, testing, and validation sets. The sample images are displayed for quick visual inspection.

# MultiMNIST Classification using CNN


### 1. Defining the CNN Model
A CNN model is defined using PyTorch's `nn.Module` class. It consists of convolutional layers followed by max-pooling layers for feature extraction and fully connected layers for classification. The model is initialized with the number of classes based on the training dataset.

### 2. Loss and Optimizer
The model is configured with a loss function, `nn.CrossEntropyLoss`, and an optimizer, `optim.Adam`. This combination is commonly used for classification tasks.

### 3. Data Loading
The training, validation, and test datasets are loaded using `DataLoader` from PyTorch. They are organized into batches for efficient training.

### 4. Training Loop
The model is trained for a specified number of epochs (`num_epochs`). In each epoch, it iterates through the training dataset, computes loss, and updates the model's parameters. The loss is backward-propagated to adjust the model's weights.

### 5. Validation
After each epoch, the model is evaluated on the validation dataset. The validation loss and accuracy are calculated. The model is set to evaluation mode, and no gradients are computed during this phase.

### 6. Testing
After training, the model is evaluated on the test dataset to assess its generalization performance. Test loss and accuracy are calculated.

### 7. Results
The code prints validation loss and accuracy for each epoch and concludes with test loss and accuracy. These metrics provide insights into the model's performance.


# Permuted MNIST Dataset with Image Plotting


### 1. Permute MNIST
A `permute_mnist` function is defined to shuffle the pixel order within each image in the MNIST dataset. The permutation is applied to all images, ensuring the model cannot rely on fixed pixel positions during training.

### 2. Plot Permuted and Original Images
A function `plot_permuted_and_original_images` is created to display a side-by-side comparison of permuted and original MNIST images. It takes a dataset of permuted images and the original MNIST dataset as input.

### 3. Example Usage
- The MNIST dataset is loaded using PyTorch, and images are converted to tensors.
- The `permute_mnist` function is applied to create a permuted version of MNIST, and the result is stored in `permuted_mnist`.
- The `plot_permuted_and_original_images` function is called to display a few samples of permuted and original images with corresponding labels.


# CNN Model for MNIST Classification


#### CNN Architecture

- A basic CNN model is defined with two convolutional layers (`conv1` and `conv2`) and two fully connected layers (`fc1` and `fc2`).
- The model architecture:
  - Input: 1 channel images (grayscale)
  - Convolutional layers with ReLU activation and max-pooling.
  - Flattening the feature maps and connecting to fully connected layers.
  - Final layer with 10 output units for classifying digits.
  
#### Training Loop

- The model is initialized, and the loss function (Cross-Entropy) and optimizer (Adam) are defined.
- The training loop runs for a specified number of epochs (10 in this case).
- For each epoch, the model is set to training mode, and the training dataset is used to update the model's weights.
- Loss is calculated using the defined loss function, and the gradients are computed and used to update the model's parameters.
  
#### Validation and Testing

- After each epoch, the model is switched to evaluation mode, and the validation dataset is used to evaluate its performance.
- The accuracy is calculated by comparing the model's predictions to the true labels.
- The results, including validation loss and accuracy, are printed after each epoch.
- Finally, the model is tested on a separate test dataset, and the test accuracy is reported.

# Training an MLP (Multi-Layer Perceptron) on Permuted MNIST


### MLP Model

- An MLP model is defined with three layers: an input layer, a hidden layer, and an output layer.
- The model architecture:
  - Input size: 28x28 (flattened)
  - Hidden layer size: 128
  - Output size: 10 (for classifying 10 digits)
- The ReLU activation function is used between layers to introduce non-linearity.

### Model Training

- The model is initialized, and the loss function (Cross-Entropy) and optimizer (Adam) are defined.
- The training loop runs for a specified number of epochs (10 in this case).
- For each epoch, the model is set to training mode, and the training dataset is used to update the model's weights.
- Loss is calculated using the defined loss function, and gradients are computed and used to update the model's parameters.
- Training accuracy and loss are computed for each epoch and printed.

### Validation

- After each training epoch, the model is switched to evaluation mode.
- The validation dataset is used to evaluate the model's performance, including loss and accuracy.

### Testing

- After training and validation, the model's performance is evaluated on a separate test dataset.
- Test accuracy is calculated and reported, providing insight into how well the model generalizes to unseen data.

# AI usage

AI was used in the entire assignment throughly, it is very hard to classify some parts of the code as human or AI as I would often use CoPilot and CHATGPT for debugging, automatically changing to better variable names, vectorisation and finding plausible errors and hints towards the actual solution.