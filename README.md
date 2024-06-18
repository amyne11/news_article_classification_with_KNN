# News Article Classification with k-NN (K neearest neighbours).

This Jupyter notebook is dedicated to classifying news articles into four distinct classes using the k-Nearest Neighbors (k-NN) algorithm. The dataset includes 800 Reuters newswire articles distributed evenly across four classes.

## Dataset

The dataset contains articles from four classes:
- "earn" (0)
- "crude" (1)
- "trade" (2)
- "interest" (3)

Each class includes 200 articles, with each article characterized by word occurrences from a vocabulary of 6428 words.

## Features

- **Sparse Matrix Representation**: The dataset is stored as a sparse matrix to efficiently handle the large, mostly empty data matrix.
- **k-NN Classifier**: Implements both Euclidean and cosine distance measures to classify the articles.
- **Multiple Experiments**: Conducts experiments with different settings for the number of neighbors and distance measures.

## Dependencies

```bash
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from sklearn.metrics import pairwise_distances as cdist
```

## Usage

### Setup
- Clone the repository or download the Jupyter notebook.
- Ensure you have the required Python packages installed.

### Execution
- Open the notebook in a Jupyter environment.
- Run the cells sequentially to observe preprocessing, model training, and evaluation steps.

## Experiments

The notebook includes several experiments:
- **Basic k-NN Classification**: Tests k-NN classification using different distance metrics and calculates accuracy statistics.
- **Error Analysis**: Uses error bar plots to illustrate the classifier's performance across a range of neighbor values.
- **Confusion Matrix**: Evaluates the model's detailed performance with a confusion matrix.

## Contributing

Contributions to enhance the classifier or extend the dataset are welcome. Please fork the repository and submit a pull request with your enhancements.

