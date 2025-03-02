# Advanced NLP and From-Scratch Model Development

This repository brings together several projects that explore advanced techniques in natural language processing, deep learning, and from-scratch model development. The projects cover a range of methods—from fine-tuning large language models using parameter-efficient methods and quantizing models for efficient inference, to building Transformers, RNNs, and classical neural networks for various tasks. Additionally, this document includes comprehensive guides on building models from scratch for classification, regression, image classification, and denoising tasks.

---

## Table of Contents

- [Advanced Natural Language Processing Projects](#advanced-natural-language-processing-projects)
  - [Transformers from Scratch](#transformers-from-scratch)
    - [Data Preprocessing and Tokenization](#data-preprocessing-and-tokenization)
    - [Transformer Model for Seq2Seq Learning](#transformer-model-for-seq2seq-learning)
    - [Training and Evaluation Pipeline](#training-and-evaluation-pipeline)
    - [Analysis of Performance and Positional Encodings](#analysis-of-performance-and-positional-encodings)
  - [PEFT from Scratch: GPT-2 Text Summarization](#peft-from-scratch-gpt-2-text-summarization)
    - [Introduction](#introduction)
    - [Dataset & Requirements](#dataset--requirements)
    - [Methods](#methods)
      - [Standard Fine-Tuning](#standard-fine-tuning)
      - [Soft Prompt Tuning](#soft-prompt-tuning)
      - [LoRA Fine-Tuning](#lora-fine-tuning)
    - [Implementation Details](#implementation-details)
    - [Results & Conclusion](#results--conclusion)
  - [Quantization from Scratch: GPT-2 Small](#quantization-from-scratch-gpt-2-small)
    - [Introduction](#introduction-1)
    - [Quantization Techniques](#quantization-techniques)
      - [Whole-Model Quantization](#whole-model-quantization)
      - [Selective Quantization](#selective-quantization)
      - [Bitsandbytes & NF4 Quantization](#bitsandbytes--nf4-quantization)
    - [Results, Graphs & Analysis](#results-graphs--analysis)
  - [RNN, N-Gram, Decoder from Scratch](#rnn-n-gram-decoder-from-scratch)
    - [Test Perplexity Scores Comparison](#test-perplexity-scores-comparison)
    - [Training Analysis](#training-analysis)
    - [Conclusion](#conclusion-1)
- [From-Scratch Development of Various Models and Techniques](#from-scratch-development-of-various-models-and-techniques)
  - [Multilayer Perceptron (MLP)](#multilayer-perceptron-mlp)
    - [Dataset Analysis and Preprocessing](#dataset-analysis-and-preprocessing)
    - [Model Building from Scratch](#model-building-from-scratch)
    - [Model Training & Hyperparameter Tuning](#model-training--hyperparameter-tuning)
    - [Evaluating Model](#evaluating-model)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
    - [Data Visualization and Preprocessing](#data-visualization-and-preprocessing)
    - [Model Building](#model-building)
    - [Hyperparameter Tuning and Evaluation](#hyperparameter-tuning-and-evaluation)
    - [Model Evaluation and Analysis](#model-evaluation-and-analysis)
    - [Train on Noisy Dataset](#train-on-noisy-dataset)
  - [AutoEncoders](#autoencoders)
    - [Denoising with AutoEncoders](#denoising-with-autoencoders)
  - [Ensemble Learning](#ensemble-learning)
    - [Bagging](#bagging)
    - [Stacking](#stacking)
    - [Random Forest vs Boosted Trees](#random-forest-vs-boosted-trees)
  - [Regression and Classification Tasks](#regression-and-classification-tasks)
    - [Wine Quality Dataset (Classification)](#wine-quality-dataset-classification)
    - [Boston Housing Dataset (Regression)](#boston-housing-dataset-regression)
    - [MNIST Dataset (Image Classification)](#mnist-dataset-image-classification)

---

## Advanced Natural Language Processing Projects

### Transformers from Scratch

#### Data Preprocessing and Tokenization
- **Objective:** English-to-French translation.
- **Key Steps:**
  - **File Paths Configuration:** Define paths for train, dev, and test datasets.
  - **Tokenization Functions:** Convert sentences to lower-case tokens.
  - **Vocabulary Building:** Generate vocabularies with special tokens (`<unk>`, `<pad>`, `<sos>`, `<eos>`).
  - **Indices Conversion & Padding:** Transform tokenized sentences into fixed-length index sequences.

#### Transformer Model for Seq2Seq Learning
- **Features:**
  - **Multi-Head Attention:** Splits embeddings to learn diverse representations.
  - **Encoder & Decoder Architecture:**
    - **Encoder:** Stacks multi-head attention and feed-forward layers with residual connections and layer normalization.
    - **Decoder:** Employs masked self-attention and cross-attention for generating target sequences.
  - **Final Linear Projection:** Transforms decoder outputs into vocabulary logits for token prediction.

#### Training and Evaluation Pipeline
- **Core Components:**
  - **Training Loop:** Iterates through epochs updating model weights using cross-entropy loss and OneCycleLR scheduler.
  - **Translation & BLEU Evaluation:** Translates sentences and assesses quality using BLEU scores.
  - **Results Saving:** Stores BLEU scores for further analysis.

#### Analysis of Performance and Positional Encodings
- **Insights:**
  - **Training Loss & BLEU Trends:** Rapid initial improvement, later plateauing—hyperparameter tuning is suggested.
  - **Positional Encodings:** Utilizes sinusoidal functions to effectively capture token order.
  - **Configuration Impact:** Evaluates different model dimensions, layers, heads, and dropout rates on performance.

---

### PEFT from Scratch: GPT-2 Text Summarization

#### Introduction
- **Objective:** Fine-tune GPT-2 for text summarization using the CNN/DailyMail dataset.
- **Strategies Explored:**
  - **Standard Fine-Tuning:** Adjust full or partial model weights.
  - **Soft Prompt Tuning:** Freeze base model; train additional prompt embeddings.
  - **LoRA Fine-Tuning:** Insert low-rank adaptation matrices to reduce trainable parameters.

#### Dataset & Requirements
- **Dataset:** CNN/DailyMail (news articles and highlights)
- **Dependencies:** Python 3.x, Transformers, Datasets, Evaluate, TRL, PEFT, PyTorch, NumPy  
  Installation command:
  ```bash
  pip install transformers datasets evaluate trl peft torch numpy
  ```

#### Methods

##### Standard Fine-Tuning
- Train the entire GPT-2 model (or parts) by adjusting weights for summarization.

##### Soft Prompt Tuning
- Freeze GPT-2 weights; learn additional prompt embeddings to guide text generation.

##### LoRA Fine-Tuning
- Integrate low-rank adaptation matrices into model layers, achieving parameter efficiency.

#### Implementation Details
- **Scripts Included:**
  - **finetuning.py:** Manages dataset tokenization, selective layer freezing, and training with ROUGE evaluation.
  - **prompt.py:** Implements soft prompt tuning by adding trainable prompt embeddings.
  - **lora.py:** Applies LoRA using PEFT, merging inputs and targets effectively.

#### Results & Conclusion
- The training procedures and evaluation (via ROUGE scores) reveal a trade-off between computational efficiency and performance. Soft prompt and LoRA methods emerge as effective parameter-efficient alternatives.

---

### Quantization from Scratch: GPT-2 Small

#### Introduction
- **Objective:** Apply quantization techniques to reduce GPT-2 Small model size and speed up inference on the WikiText-2 dataset while maintaining acceptable performance.

#### Quantization Techniques

##### Whole-Model Quantization
- Convert all weights from 32-bit/16-bit floating point to 8-bit integers using calculated scale factors and zero-points.

##### Selective Quantization
- Focus on quantizing key components such as:
  - **MLP Layers:** Only feed-forward network parameters.
  - **Decoder Layers:** Selectively quantize based on layer indices.

##### Bitsandbytes & NF4 Quantization
- **8-bit Quantization:** Utilize Bitsandbytes for 8-bit precision loading.
- **4-bit Quantization:** Reduce precision further to 4 bits with double quantization.
- **NF4 Quantization:** Employ non-linear 4-bit quantization to better retain essential weight details.

#### Results, Graphs & Analysis
- Comparative metrics include model size, perplexity, and inference latency. Graphs illustrate the trade-offs between quantization level, performance degradation, and speed improvement.

---

### RNN, N-Gram, Decoder from Scratch

#### Test Perplexity Scores Comparison
- **5-gram Model:** 305.28
- **LSTM Model:** 198.35
- **Decoder Model:** 194.3

#### Training Analysis
- **Observations:**
  - **Loss Trends:** LSTM shows a smooth decline, while the 5-gram model fluctuates.
  - **Perplexity Trends:** Despite a slower start, the Decoder model eventually achieves the lowest perplexity.

#### Conclusion
- The Decoder model, leveraging self-attention mechanisms, outperforms traditional n-gram and LSTM models in terms of perplexity, especially for short sentences.

---

## From-Scratch Development of Various Models and Techniques

### Multilayer Perceptron (MLP)

#### Dataset Analysis and Preprocessing
- **Dataset Description:**  
  - Summary statistics (mean, standard deviation, min, max) and distribution graphs.
- **Data Partitioning:**  
  - Splitting into training, validation, and test sets.
  - Data normalization and standardization.
  - Handling missing or inconsistent values.

#### Model Building from Scratch
- **Implementation:**
  - Create a class with configurable parameters (learning rate, activation functions, optimizers, number of hidden layers, neurons).
  - Methods for forward propagation, backpropagation, and training.
  - Include activation functions: Sigmoid, Tanh, ReLU.
  - Implement various optimization algorithms: Stochastic Gradient Descent (SGD), Batch Gradient Descent, Mini-Batch Gradient Descent.

#### Model Training & Hyperparameter Tuning
- **Experimentation with Weights & Biases (W&B):**
  - Logging metrics and experimenting with hyperparameters (learning rate, epochs, neurons, etc.).
  - Tracking performance using accuracy, F1 score, precision, and recall.

#### Evaluating Model
- **Test Set Evaluation:**
  - Generate and print a classification report.
  - Compare performance against a logistic regression baseline.

---

### Convolutional Neural Network (CNN)

#### Data Visualization and Preprocessing
- **Visualization:**
  - Graphs depicting label distributions and sample images.
  - Assessing class imbalances.
- **Data Partitioning:**
  - Splitting data into training, validation, and test sets.
  - Visualizing feature maps for insights.

#### Model Building
- **Architecture:**
  - Build a CNN incorporating convolutional layers, pooling layers, dropout, and fully connected layers.
  - Implement a baseline CNN and display feature maps.

#### Hyperparameter Tuning and Evaluation
- **Using W&B:**
  - Experiment with different architectures and hyperparameters.
  - Log training and validation losses, accuracy, confusion matrices, and class-specific metrics.

#### Model Evaluation and Analysis
- **Test Set Evaluation:**
  - Report overall accuracy and per-class accuracy.
  - Generate a detailed classification report and confusion matrix.

#### Train on Noisy Dataset
- **Noisy MNIST Dataset:**
  - Train the optimal CNN model on noisy data.
  - Report training and validation losses and scores.
  - Evaluate on test data and produce a classification report.

---

### AutoEncoders

#### Denoising with AutoEncoders
- **Implementation:**
  - Create an Autoencoder class focused on denoising.
  - Visualize feature space changes before and after denoising.
  - Train on the de-noised dataset and compare validation/training scores.

---

### Ensemble Learning

#### Bagging
- **Methodology:**
  - Implement a bagging approach with configurable parameters: base estimator, number of estimators, sampling method (bootstrap), and voting mechanism.
  - Train ensemble models and compare performance using heatmaps and histograms.

#### Stacking
- **Methodology:**
  - Build a stacking ensemble with Level-0 estimators and a Level-1 estimator.
  - Evaluate ensemble performance and compare training times with bagging.

#### Random Forest vs Boosted Trees
- **Comparative Analysis:**
  - Train a Random Forest classifier/regressor.
  - Compare results with boosted decision trees.
  - Analyze model mistakes and feature importance similarities.

---

### Regression and Classification Tasks

#### Wine Quality Dataset (Classification)
- **Multinomial Logistic Regression:**
  - Develop a logistic regression model using cross-entropy loss and gradient descent.
  - Tune hyperparameters using validation sets and W&B.
  - Evaluate on the test set with a comprehensive classification report.
- **MLP Classification:**
  - Build an MLP classifier from scratch.
  - Experiment with various activation functions and optimizers.
  - Log performance using W&B and compare against logistic regression results.

#### Boston Housing Dataset (Regression)
- **MLP Regression:**
  - Construct an MLP model for regression.
  - Experiment with different activation functions and optimization techniques.
  - Evaluate using metrics such as MSE, RMSE, and R-squared on the test set.

#### MNIST Dataset (Image Classification)
- **CNN for Image Classification:**
  - Develop a CNN using PyTorch for classifying MNIST images.
  - Tune hyperparameters via W&B.
  - Report accuracy and other evaluation metrics on the test set.
- **AutoEncoders for Denoising:**
  - Implement an Autoencoder to denoise the MNIST dataset.
  - Train the best CNN model on the denoised data and compare performance with noisy data results.

---

### Additional Notes

- **Project Organization:**  
  Each project is organized within its respective folder:
  - **PEFT:** Contains scripts for standard, soft prompt, and LoRA fine-tuning.
  - **Quantization:** Houses quantization implementations and evaluation graphs.
  - **RNN:** Includes models and training analysis visuals.
  - **Transformers:** Contains data preprocessing, model definitions, and training pipelines.
- **Comprehensive Reports:**  
  Refer to the detailed reports in sub-folders for additional insights into parameter-efficient fine-tuning, quantization, and Transformer implementations.

---