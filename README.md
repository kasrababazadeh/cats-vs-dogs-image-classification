# Cats vs. Dogs Image Classification

## Overview

This project focuses on classifying images of cats and dogs using Convolutional Neural Networks (CNNs). Inspired by the classic [Kaggle Dogs vs. Cats competition](https://www.kaggle.com/c/dogs-vs-cats), this notebook demonstrates the process of loading image data, preprocessing it, training a CNN, and evaluating model performance.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

To get started with this project, you'll need to set up your environment. The following dependencies are required:

- Python 3.x
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- tqdm

You can install the required packages using pip:

```bash
pip install torch torchvision numpy pandas matplotlib tqdm
```

## Usage

1. **Clone this repository:**

   ```bash
   git clone https://github.com/kasrababazadeh/cats-vs-dogs-classification.git
   cd cats-vs-dogs-classification
   ```

2. **Download the Cats vs. Dogs dataset:**

The dataset can be downloaded from this link. Make sure to place the unzipped dataset in the CATS_DOGS directory.

3. **Open the Jupyter Notebook:**

   ```bash
   jupyter notebook Cats_vs_Dogs_Classification.ipynb
   ```

4. **Run the notebook cells sequentially to train and evaluate the model.**

## Dataset

The dataset consists of two folders, `CAT` and `DOG`, each containing images of their respective animals. The notebook provides functions to preprocess these images and prepares them for training and testing.

## Model Architecture

This project implements a simple CNN architecture with the following layers:

- Two convolutional layers followed by max pooling
- Three fully connected layers
- ReLU activations

The notebook also demonstrates how to leverage a pretrained model (AlexNet) for transfer learning.

## Training the Model

The training process is done in epochs, and the model is evaluated on a separate test dataset. The notebook includes visualizations of the training loss and accuracy.

## Results

Upon completion, the notebook displays the final training and testing accuracy of the model. Sample predictions for new images can also be viewed, along with the respective predictions from the AlexNet model.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- Kaggle for the Cats vs. Dogs dataset
- PyTorch for the deep learning framework
- OpenAI for the inspiration and tools

Feel free to reach out if you have any questions or feedback!
