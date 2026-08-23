# TextCNN Project

## Overview

This project implements a **Text Convolutional Neural Network (TextCNN)** for text classification tasks using TensorFlow. TextCNN is a powerful neural network architecture that uses convolutional layers to extract local features from text data.

## Project Description

TextCNN is a deep learning model designed to classify text by:
- **Embedding Layer**: Converts text tokens into dense vector representations
- **Convolutional Layers**: Applies multiple filters of varying sizes to extract features at different granularities
- **Max Pooling**: Captures the most important features from each filter
- **Fully Connected Layer**: Combines all extracted features for final classification
- **Dropout Regularization**: Prevents overfitting by randomly dropping units during training
- **L2 Regularization**: Further reduces overfitting by penalizing large weights

## Architecture

The model architecture includes:

```
Input (Sequences) 
    ↓
Embedding Layer (Converts tokens to vectors)
    ↓
Convolutional Filters (Multiple filter sizes for feature extraction)
    ↓
Max Pooling (Extracts important features)
    ↓
Concatenation (Combines all pooled outputs)
    ↓
Dropout (Regularization)
    ↓
Fully Connected Layer (Classification)
    ↓
Output (Class predictions)
```

## Key Components

### TextCNN Class

The main class that constructs the TextCNN model with the following parameters:

- **sequence_length**: Maximum length of input sequences
- **num_classes**: Number of output classes
- **vocab_size**: Size of the vocabulary
- **embedding_size**: Dimensionality of word embeddings
- **filter_sizes**: List of filter window sizes (e.g., [3, 4, 5])
- **num_filters**: Number of filters for each filter size
- **l2_reg_lambda**: L2 regularization coefficient (default: 0.0)

### Model Layers

1. **Placeholders**: Input, output, and dropout keep probability
2. **Embedding**: Transforms input tokens into dense vectors
3. **Convolution + ReLU**: Extracts features using convolutional filters
4. **Max Pooling**: Captures the most significant features
5. **Dropout**: Regularization to prevent overfitting
6. **Output Layer**: Fully connected layer with softmax cross-entropy loss
7. **Accuracy Metric**: Computes classification accuracy

## Dependencies

The project requires the following libraries:

```python
tensorflow
numpy
scikit-learn
```

Install dependencies using:
```bash
pip install tensorflow numpy scikit-learn
```

## Usage

```python
from your_module import TextCNN

# Initialize the model
model = TextCNN(
    sequence_length=56,
    num_classes=2,
    vocab_size=5000,
    embedding_size=100,
    filter_sizes=[3, 4, 5],
    num_filters=100,
    l2_reg_lambda=0.001
)

# Use model.input_x, model.input_y, model.dropout_keep_prob for training
# Use model.predictions for inference
# Use model.accuracy to evaluate performance
```

## Features

- ✅ Multi-filter convolutional architecture for robust feature extraction
- ✅ Flexible configuration for different text classification tasks
- ✅ Built-in accuracy computation and loss calculation
- ✅ L2 regularization to prevent overfitting
- ✅ Dropout support for additional regularization
- ✅ Compatible with TensorFlow's computational graph execution

## Model Metrics

The model includes evaluation metrics:
- **Loss**: Softmax cross-entropy loss with L2 regularization
- **Accuracy**: Classification accuracy on test data
- **Predictions**: Argmax predictions for classification

## Notes

- The model uses TensorFlow 1.x style placeholders and sessions
- Word embeddings are randomly initialized and learned during training
- The convolutional filters extract n-gram features from the text
- Max pooling ensures translation invariance in feature extraction

## License

This project is part of the KG_projects repository.

## Author

Created by fgbah

---

For more information on TextCNN, refer to the original paper:
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim (2014)
