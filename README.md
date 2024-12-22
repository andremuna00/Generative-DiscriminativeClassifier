# 🖊️ MNIST Handwritten Digit Classifier Project

## 🔍 Overview

This project focuses on training a model to classify handwritten digits using the MNIST dataset. The dataset comprises 70,000 grayscale images of handwritten digits, each of size 28×28 pixels. These images are divided into:

- Training Set: 60,000 images.
- Test Set: 10,000 images.

Each image is represented as a 784-dimensional vector of real values between 0 (black) and 1 (white). The task is to use various machine learning techniques to train and evaluate classifiers for this dataset.


## ⚙️ Implementation Details

### Methods Used

The following classifiers were implemented and tested:

1. Support Vector Machine (SVM):
    - Linear kernel.
    - Polynomial kernel of degree 2.
    - Radial Basis Function (RBF) kernel.
2. Random Forests:
    - Implemented using the scikit-learn library.
3. Naive Bayes Classifier:
    - Custom implementation.
4. k-Nearest Neighbors (k-NN):
    - Custom implementation.

### Cross-Validation

To optimize hyperparameters for each classifier, 10-fold cross-validation is applied.

## 🚀 Usage Instructions

### Requirements

Install the necessary Python libraries:
```bash
pip install numpy scikit-learn
```


### Dataset Preparation

Ensure the MNIST dataset is available in the required format. Use libraries like tensorflow or sklearn to load the data if needed:

```bash
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
data = mnist.data
labels = mnist.target
```

## 📊 Results

The performance of each classifier is evaluated based on:

- Accuracy on the test set.
- Precision, recall, and F1-score metrics.


## 📂 References

- MNIST Dataset
- scikit-learn Documentation: [scikit-learn.org](https://scikit-learn.org/stable/)

