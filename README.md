# Meta-Learning (MAML) Model with ResNet-50 Backbone for Image Classification


This repository contains code for a Meta-Learning (MAML) model implemented using PyTorch, with a ResNet-50 backbone for image classification tasks. The model is designed to adapt quickly to new tasks through meta-learning principles.
![download](https://github.com/user-attachments/assets/a46f5176-a696-441e-85f0-30dd8e02c160)
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

## Usage

### Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- seaborn

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Training

To train the MAML model on your dataset, follow these steps:

1. Prepare your dataset and ensure it is structured correctly.
2. Set the dataset paths in the script.
3. Adjust hyperparameters such as learning rate (`lr`), number of epochs, and batch size as needed.
4. Run the training script:

   ```bash
   python train.py
   ```

### Evaluation

After training, evaluate the model's performance:

1. Load trained model weights.
2. Evaluate metrics like accuracy, precision, recall, and F1-score using test data.
3. Visualize results using confusion matrices and sample correctly/misclassified images.

### Example Usage

```python
python train.py  # To train the model
python evaluate.py  # To evaluate the trained model
```

## Refrences
https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset?select=Vegetable+Images
https://www.researchgate.net/publication/352846889_DCNN-Based_Vegetable_Image_Classification_Using_Transfer_Learning_A_Comparative_Study
## Acknowledgments

- Inspiration and foundational concepts from Model-Agnostic Meta-Learning (MAML) research.
- PyTorch community for their powerful and flexible deep learning framework.

