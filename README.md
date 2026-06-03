# Breast Cancer Detection Assistance

## Overview
A three-phase hybrid deep learning and machine learning pipeline for classifying 
breast cancer histopathology images as benign or malignant. Built on the BreakHis 
dataset (9,109 microscopic images at 40x magnification, stained with H&E), this 
project combines GLCM texture feature extraction with CNN architectures and 7 ML 
classifiers.

## Key Results
| Model | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| Sequential CNN + SVM (best hybrid) | **95.05%** | 0.9167 | 0.9943 | 0.9491 |
| Sequential CNN + Logistic Regression | 95.05% | 0.9167 | 0.9943 | 0.9492 |
| MobileNet + SVM | 94.95% | 0.9123 | 0.9981 | 0.9481 |
| MobileNet (Phase 2, GLCM) | 94.75% | 0.9228 | 0.9831 | 0.9880 |
| Sequential CNN (Phase 2, GLCM) | 94.84% | 0.9175 | 0.9915 | 0.9875 |
| MobileNet (Phase 1, raw images) | 93.32% | — | — | — |

## Dataset
- **BreakHis** — 9,109 breast tumour histopathology images
- Labels: Benign vs Malignant (4 subtypes each)
- Split: 6,406 training / 791 validation / 712 test
- Class imbalance handled via upsampling (4,404 malignant → matched with 4,404 
  benign in training set)
- Data augmentation: rotations at 0°, 45°, 90°, 135°

## Project Structure
- `phase1_transfer_learning.py` — ResNet50 and MobileNet trained on raw images
- `phase2_glcm_cnn.py` — GLCM feature extraction + Sequential CNN and MobileNet
- `phase3_hybrid_models.py` — CNN/MobileNet features fed into 7 ML classifiers
- `glcm_extraction.py` — GLCM feature extractor (6 features × 4 directions = 24 features)

## Methodology

### Phase 1 — Transfer Learning on Raw Images
- Trained ResNet50 and MobileNet on BreakHis images resized to 256×256
- 20 epochs each; MobileNet achieved higher accuracy (93.32%) and was carried forward

### Phase 2 — GLCM Feature Extraction + CNN Training
- Extracted 24 GLCM features per image: Dissimilarity, Correlation, Homogeneity, 
  Contrast, ASM, and Energy — each computed at 4 orientations (0°, 45°, 90°, 135°)
- Feature matrix size: 7,909 × 25
- Designed a custom 17-layer Sequential CNN alongside MobileNet
- Both trained for 30 epochs on GLCM features; Sequential CNN achieved 94.84% accuracy

### Phase 3 — Hybrid CNN + ML Models
- Features extracted from trained CNN and MobileNet models used as input to 7 ML 
  classifiers: SVM, Decision Tree, Logistic Regression, KNN, Random Forest, 
  Gradient Boosting, XGBoost
- CNN-extracted features consistently outperformed MobileNet features across all 
  classifiers
- Best overall: Sequential CNN + SVM/LR/KNN/RF/DT/XGB all achieved 95.05% accuracy

## GLCM Features Extracted
1. Dissimilarity — measures contrast between neighbouring pixels
2. Correlation — linear dependency between pixel pairs
3. Homogeneity — uniformity of gray-level distribution
4. Contrast — local variation in gray levels
5. ASM (Angular Second Moment) — overall uniformity
6. Energy — square root of ASM

## Tools & Technologies
**Language:** Python  
**Deep Learning:** TensorFlow, Keras (MobileNet, ResNet50, Sequential CNN)  
**Machine Learning:** Scikit-learn (SVM, KNN, LR, DT, RF, Gradient Boosting), XGBoost  
**Feature Extraction:** GLCM (skimage.feature)  
**Dataset:** BreakHis Histopathological Image Dataset  
**Metrics:** Accuracy, Precision, Recall, F1-score, AUC, Loss  

## Collaborators
- Vineet Naren Belagod
- Anjana Guru Prasad
- Shruti Kulkarni
- Vaishnavi Bhangennavar

*Undergraduate final year project — BMS Institute of Technology and Management, 2022–23*
