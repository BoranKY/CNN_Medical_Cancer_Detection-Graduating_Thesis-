# Cancer Detection from Radiological Images

## Overview
This project presents a system for detecting tumors from radiological images, specifically focusing on MR (Magnetic Resonance Imaging) and PET (Positron Emission Tomography) scans. The system leverages Convolutional Neural Networks (CNN) to analyze and classify images from the chest region to assist healthcare professionals in identifying potential cases of cancer or suggesting further review in cases of possible misdiagnosis.

## Objectives
- To develop a deep learning model that can accurately detect tumors in MR and PET images.
- To provide a tool that assists healthcare workers in making more accurate diagnoses.
- To compare the performance of models trained on MR images only, PET images only, and combined MR and PET images.

## Project Structure
- **data/**: Contains the datasets used for training and testing. The datasets include images categorized into positive (tumor present) and negative (no tumor) cases.
- **notebooks/**: Jupyter notebooks used for data preprocessing, model training, and evaluation.
- **src/**: Source code for the model definition, training, and evaluation scripts.
- **results/**: Contains the results of the model evaluations, including metrics and confusion matrices.
- **README.md**: This file, providing an overview and instructions for the project.

## Models and Methods
### Data Collection and Preprocessing
- **MR and PET Images**: The dataset includes images of the chest region. MR images provide high-resolution details of tissues, while PET images highlight metabolic processes.
- **Data Augmentation**: Techniques such as rotation, cropping, and normalization were applied to increase the dataset's variability and improve model robustness.

### CNN Architecture
- **ResNet 34**: The primary model used is ResNet 34, a deep residual network known for its effectiveness in image classification tasks. This architecture helps in addressing the vanishing gradient problem and allows for the training of very deep networks.

### Training and Evaluation
- The model was trained using cross-entropy loss and optimized with the Adam optimizer. Different evaluation metrics, including accuracy, precision, recall, and the ROC-AUC score, were used to assess the model's performance.

## Results
- The combined MR and PET image model showed the highest performance with better classification metrics compared to models trained on only MR or only PET images.
- Detailed results and performance metrics for each dataset can be found in the `results/` directory.


