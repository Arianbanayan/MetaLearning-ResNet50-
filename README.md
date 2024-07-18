# Meta-Learning (MAML) Model with ResNet-50 Backbone for Image Classification


This repository contains code for a Meta-Learning (MAML) model implemented using PyTorch, with a ResNet-50 backbone for image classification tasks. The model is designed to adapt quickly to new tasks through meta-learning principles.


![dataset-cover (2)](https://github.com/user-attachments/assets/10c11182-2a42-4b62-a978-703b065c833b)

## Overview

Meta-Learning, specifically Model-Agnostic Meta-Learning (MAML), is a technique that enables a model to learn new tasks quickly with minimal data. This implementation uses a ResNet-50 backbone pretrained on ImageNet to extract features from images, followed by a linear classifier for classification into multiple classes.

### Components

1. **Device Assignment:**
   - Determines whether to use GPU (`cuda:0`) or CPU (`cpu`) based on availability.

2. **ResNet Backbone (`ResNetBackbone` class):**
   - Initializes ResNet-50 and removes its final fully connected layer (`fc`) for feature extraction.

3. **MAML Model (`MAMLModel` class):**
   - Extends the ResNet backbone with a linear classifier for classifying features into `output_dim` classes.

4. **Model Training Setup:**
   - Initializes the MAML model, criterion (Cross-Entropy Loss), and optimizer (Adam) for training.

5. **Evaluation and Metrics:**
   - Evaluates the trained model using metrics such as accuracy, precision, recall, and F1-score.
   - Visualizes model performance with confusion matrices and correctly/misclassified images.


## Refrences
https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset?select=Vegetable+Images
https://www.researchgate.net/publication/352846889_DCNN-Based_Vegetable_Image_Classification_Using_Transfer_Learning_A_Comparative_Study


