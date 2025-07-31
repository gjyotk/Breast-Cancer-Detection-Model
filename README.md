# Breast Cancer Detection from Ultrasound Images
**Project Overview**

This repository implements a complete pipeline for breast cancer detection using ultrasound images from the publicly available BUS-BRA dataset. It combines fine-tuned segmentation and classification models with a user-friendly web interface to facilitate both research and practical application. The goal is to create an end-to-end Computer-Aided Diagnostic (CAD) tool that segments regions of interest in ultrasound scans and classifies them as benign or malignant.


### Dataset

* **Name**: BUS-BRA (Breast Ultrasound Images) Dataset (paper available [here](https://pubmed.ncbi.nlm.nih.gov/37937827/))
* **Source**: National Institute of Cancer, Rio de Janeiro, Brazil
* **Content**:

  * 1,875 anonymized ultrasound images
  * 1,064 female patients
  * Acquired with four different ultrasonography scanners


### Technology Stack

* **Languages & Frameworks**: Python, JavaScript, HTML, CSS
* **Backend**: FastAPI
* **Frontend**: Vanilla JS, HTML, CSS
* **Deep Learning**: TensorFlow & Keras
* **Image Processing**: OpenCV
* **Data Handling**: NumPy, Pandas
* **Machine Learning Utils**: scikit-learn
* **Visualization**: Matplotlib, Seaborn

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/gjyotk/Breast-Cancer-Detection-Model.git
   cd web-app
   cd backend
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download and prepare the dataset and models**

   ```bash
   # Place the datasets and models in the correct folder
   ```

5. **Start the backend server**

   ```bash
   uvicorn app.main:app --reload
   ```

6. **Open the frontend**

   * Navigate to `frontend/index.html` in your browser


## Methodology

### 1. Preprocessing

* **Resizing & Normalization**: All ultrasound images resized to `(224 × 224)` pixels, pixel values scaled to `[0, 1]`.
* **Augmentation**: Advanced techniques via OpenCV (rotations, flips, brightness/contrast variations).

### 2. Segmentation with U-Net

* **Architecture**: Standard U-Net implementation focused on biomedical image segmentation.
* **Loss**: Dice Loss to optimize region overlap.
* **Metric**: Dice Coefficient for quantitative performance.

### 3. Handling Imbalanced Data

* **SMOTE**: Synthetic Minority Over-sampling Technique to balance benign vs. malignant classes.
* **Implementation**: scikit-learn’s `SMOTE` module

### 4. Classification with ResNet50

* **Base Model**: Pre-trained on ImageNet.
* **Fine-Tuning**:

  * Freeze first 100 layers
  * Retrain remaining layers on the processed dataset
* **Metrics**:

  * Accuracy
  * AUC (Area Under the ROC Curve)
  * Precision
  * Recall
  * True Positives & False Positives


## Web Application

A lightweight web interface allows users to upload a single ultrasound image and obtain:

1. **Segmented Image**: U-Net’s output mask overlaid on the original scan.
2. **Classification Result**: Benign or Malignant prediction from the fine-tuned ResNet50.


## Results and Visualisations

<img width="4807" height="2765" alt="data-preview2" src="https://github.com/user-attachments/assets/15171b99-6e0c-457c-b91b-3cb9d5b366a3" />


<img width="910" height="329" alt="seg1" src="https://github.com/user-attachments/assets/d42e44cc-5a0b-4ea4-8836-044978ab4378" />

<img width="474" height="505" alt="training-and-validation-loss" src="https://github.com/user-attachments/assets/3c4478c5-151c-4a4a-bc69-3224a07ba9b4" />


## Future Work

* Experiment with more advanced augmentations.
* Explore explainable AI for model interpretability.
* Work on improving classification model performance.
* Deploy full-stack solution on cloud.


## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
