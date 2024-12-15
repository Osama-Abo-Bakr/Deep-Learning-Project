# Deep Learning Projects Repository

This repository contains a collection of **Deep Learning (DL)** projects, demonstrating practical implementations of cutting-edge DL techniques and architectures. Each project tackles a unique problem, ranging from image classification and object detection to natural language processing and segmentation.

---

## Table of Contents
1. [Projects Overview](#projects-overview)
2. [Installation](#installation)
3. [Project Details](#project-details)
   - [1. Breast Cancer Detection](#1-breast-cancer-detection)
   - [2. Skin Disease Detection](#2-skin-disease-detection)
   - [3. Axial MRI Detection](#3-axial-mri-detection)
   - [4. Transformer-based Translation](#4-transformer-based-translation)
4. [Technologies Used](#technologies-used)
5. [License](#license)

---

## Projects Overview

This repository includes the following deep learning projects:
1. **Breast Cancer Detection**: Detecting and segmenting cancerous tissues using CNNs and ViT.
2. **Skin Disease Detection**: Identifying skin conditions with YOLO and other DL architectures.
3. **Axial MRI Detection**: Detecting brain-related conditions from axial MRI scans using YOLO models.
4. **Transformer-based Translation**: Translating Arabic to Italian using MarianMT models.
Each project is documented with code, datasets, and results.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/DL-Projects.git
   cd DL-Projects
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install additional requirements specific to a project:
   Each project folder may include an extra `requirements.txt` file.

---

## Project Details

### 1. Breast Cancer Detection
**Description**: Detects and segments cancerous tissues in histopathology images using deep learning.  
- **Dataset**: [Camelyon17 Dataset](https://camelyon17.grand-challenge.org/).  
- **Models**: ResNet50, EfficientNet, Xception, Vision Transformers (ViT).  

**Key Features**:
- Classification and segmentation pipelines.
- Pretrained and fine-tuned DL models.
- Evaluation metrics: accuracy, IoU, and F1 score.

**Path**: `/projects/breast_cancer_detection`

---

### 2. Skin Disease Detection
**Description**: Detects and classifies skin diseases using object detection models like YOLO.  
- **Dataset**: Dermatology image datasets.  
- **Model**: YOLOv8.

**Key Features**:
- Bounding box detection for multiple skin conditions.
- Integration with preprocessing and augmentation techniques.
- Robust model evaluation.

**Path**: `/projects/skin_disease_detection`

---

### 3. Axial MRI Detection
**Description**: Uses YOLO to detect brain-related conditions in axial MRI scans.  
- **Dataset**: Custom MRI datasets.  
- **Model**: YOLOv8.

**Key Features**:
- Handles medical image processing pipelines.
- End-to-end deployment readiness.
- Performance evaluation: mAP and detection accuracy.

**Path**: `/projects/axial_mri_detection`

---

### 4. Transformer-based Translation
**Description**: Implements neural machine translation from Arabic to Italian using a pretrained MarianMT model.  
- **Dataset**: [News Commentary](https://huggingface.co/datasets/Helsinki-NLP/news_commentary).  
- **Model**: Helsinki-NLP MarianMT (opus-mt-ar-it).  

**Key Features**:
- Data preprocessing for multilingual translation.
- Custom training with BLEU evaluation.
- Streamlit app for translation demonstrations.

**Path**: `/projects/translation_ar_it`

---

## Technologies Used
- **Frameworks**: PyTorch, TensorFlow, Hugging Face
- **Models**: CNNs, YOLO, Vision Transformers, MarianMT
- **Tools**: OpenCV, Streamlit, Flask
- **Evaluation Metrics**: BLEU, IoU, mAP, F1-score
- **Programming Language**: Python
- **Visualization**: Matplotlib, Seaborn

---

## License
This repository is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing
Contributions are welcome! Follow these steps to contribute:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m "Add feature"`.
4. Push the branch: `git push origin feature-name`.
5. Open a pull request.

For queries or suggestions, feel free to open an issue.

---
