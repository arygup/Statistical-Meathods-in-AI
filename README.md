# From-Scratch Development of Various Models and Techniques

This document aims to provide a comprehensive guide on the aspects of these from scratch models, including data preprocessing, model building, training, and evaluation. 
Used the Wine Quality Dataset and Boston Housing Dataset for classification and regression tasks, respectively. Additionally, the MNIST dataset was used for image classification tasks.

## Table of Contents

1. [Multilayer Perceptron (MLP)](#multilayer-perceptron-mlp)
   - Dataset Analysis and Preprocessing
   - Model Building from Scratch
   - Model Training & Hyperparameter Tuning
   - Evaluating Model
2. [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
   - Data Visualization and Preprocessing
   - Model Building
   - Hyperparameter Tuning and Evaluation
   - Model Evaluation and Analysis
   - Train on Noisy Dataset
3. [AutoEncoders](#autoencoders)
   - Denoising with AutoEncoders
4. [Ensemble Learning](#ensemble-learning)
   - Bagging
   - Stacking
   - Random Forest vs Boosted Trees

## Multilayer Perceptron (MLP)

### Dataset Analysis and Preprocessing

1. **Dataset Description**:
   - Mean, standard deviation, min, and max values for all attributes.
   - Graph showing the distribution of various labels.

2. **Data Partitioning**:
   - Split dataset into train, validation, and test sets.
   - Normalize and standardize the data.
   - Handle missing or inconsistent data values.

### Model Building from Scratch

1. **Class Implementation**:
   - Create a class to modify and access learning rate, activation function, optimizers, number of hidden layers, and neurons.
   - Methods for forward propagation, backpropagation, and training.
   - Implement Sigmoid, Tanh, and ReLU activation functions.
   - Implement Stochastic Gradient Descent (SGD), Batch Gradient Descent, and Mini-Batch Gradient Descent algorithms.

### Model Training & Hyperparameter Tuning

1. **Using Weights & Biases (W&B)**:
   - Log and track model’s metrics.
   - Experiment with hyperparameters such as learning rate, epochs, hidden layer neurons, activation functions, and optimization techniques.
   - Report metrics: accuracy, f-1 score, precision, and recall.

### Evaluating Model

1. **Test Set Evaluation**:
   - Print classification report.
   - Compare results with logistic regression model.

## Convolutional Neural Network (CNN)

### Data Visualization and Preprocessing

1. **Data Visualization**:
   - Graph showing the distribution of various labels.
   - Visualize several samples of images from each class.
   - Check for class imbalance.

2. **Data Partitioning**:
   - Split dataset into train, validation, and test sets.
   - Function to visualize feature maps.

### Model Building

1. **Construct CNN Model**:
   - Include convolutional layers, pooling layers, dropout layers, and fully connected layers.
   - Baseline CNN with specific architecture.
   - Display feature maps and analysis.

### Hyperparameter Tuning and Evaluation

1. **Using W&B**:
   - Experiment with various architectures and hyperparameters.
   - Log training/validation loss and accuracy, confusion matrices, and class-specific metrics.

### Model Evaluation and Analysis

1. **Test Set Evaluation**:
   - Report accuracy, per-class accuracy, and classification report.
   - Visualization of model performance (e.g., confusion matrix).

### Train on Noisy Dataset

1. **Noisy MNIST Dataset**:
   - Train the best model on noisy dataset.
   - Report validation losses, validation scores, training losses, and training scores.
   - Evaluate on test data and print classification report.

## AutoEncoders

### Denoising with AutoEncoders

1. **Autoencoder Implementation**:
   - Implement Autoencoder class for denoising.
   - Visualize classes and feature space before and after denoising.
   - Train the model on the de-noised dataset.
   - Report validation and training scores.

## Ensemble Learning

### Bagging

1. **Bagging Methodology**:
   - Function for bagging with parameters: Base Estimator Model, Number of Estimators, Fraction/Number of Samples, Bootstrap, and Voting Mechanism.
   - Train ensemble models and report best-performing models.
   - Compare performance with heatmaps and side-by-side histograms.

### Stacking

1. **Stacking Methodology**:
   - Function for stacking with parameters: Level-0 estimators, Level-1 estimators, and Stacking Methodologies.
   - Train ensemble models and report best-performing models.
   - Compare accuracies and training time of Bagging and Stacking ensembles.

### Random Forest vs Boosted Trees

1. **Random Forest**:
   - Train Random Forest Classifier and Regressor.
   - Compare with Boosted Decision Trees.
   - Analyze model mistakes and feature similarity.

## Regression and Classification Tasks

### Wine Quality Dataset (Classification)

1. **Multinomial Logistic Regression**:
   - Implement a Multinomial Logistic Regression model from scratch.
   - Use cross-entropy loss and gradient descent for optimization.
   - Train the model and report metrics on the validation set.
   - Fine-tune hyperparameters using validation set and W&B logging.
   - Evaluate the model on test dataset and print classification report.

2. **Multi-Layer Perceptron Classification**:
   - Implement MLP classification from scratch.
   - Experiment with various activation functions (Sigmoid, Tanh, ReLU) and optimization techniques (SGD, Batch Gradient Descent, Mini-Batch Gradient Descent).
   - Train and tune the model using W&B.
   - Report metrics and compare with logistic regression model.

### Boston Housing Dataset (Regression)

1. **Multi-Layer Perceptron Regression**:
   - Implement MLP for regression from scratch.
   - Experiment with various activation functions and optimization techniques.
   - Train and tune the model using W&B.
   - Report metrics: MSE, RMSE, R-squared.
   - Evaluate the model on the test set.

### MNIST Dataset (Image Classification)

1. **CNN for Image Classification**:
   - Implement a CNN for image classification using PyTorch.
   - Train the model on MNIST dataset.
   - Tune hyperparameters using W&B.
   - Report accuracy and other metrics on the test set.
   - Train and evaluate on noisy MNIST dataset.

2. **AutoEncoders for Denoising**:
   - Implement an Autoencoder to denoise the MNIST dataset.
   - Train the best CNN model on the de-noised dataset.
   - Compare the performance with the noisy dataset results.

