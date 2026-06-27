# Chest X-Ray Pneumonia Diagnosis AI

A deep learning image classification model built using PyTorch and ResNet18 to automate the detection of pneumonia from medical chest X-ray images. This project demonstrates the practical application of transfer learning, handling severe data imbalance, and interpreting model performance through advanced metrics like confusion matrices.

## 🚀 Key Achievements & Performance
* **88% Overall Accuracy:** Successfully fine-tuned a pre-trained ResNet18 architecture using Transfer Learning.
* **99% Clinical Recall Rate:** The optimized model catches 385 out of 390 actual pneumonia cases, minimizing dangerous False Negatives.
* **48% Reduction in False Positives:** Engineered a custom data-balancing solution that cut misdiagnosed healthy cases nearly in half, boosting healthy lung recall from 41% to 69%.
* <p align="center">
  <img src="Screenshot 2026-06-27 101259.png" width="450" alt="Baseline Confusion Matrix">
  <br>
  <em>Figure 1: Baseline Confusion Matrix showing clear majority-class bias for Pneumonia.</em>
</p>

<p align="center">
  <img src="Screenshot 2026-06-27 102309.png" width="450" alt="Weighted Model Confusion Matrix">
  <br>
  <em>Figure 2: Optimized Weighted Model Matrix showing balanced predictions and dramatic reduction in False Positives.</em>
</p>

<p align="center">
  <img src="Screenshot 2026-06-27 104229.png" width="550" alt="AI Diagnosis Inference Example">
  <br>
  <em>Figure 3: Live inference test on a brand-new Pneumonia X-ray (99.96% Confident).</em>
</p>

---

## 🛠️ Tech Stack & Concepts Used
* **Framework:** PyTorch, Torchvision
* **Model Architecture:** ResNet18 (Pre-trained on ImageNet)
* **Optimization:** Adam Optimizer, CrossEntropyLoss
* **Hardware Acceleration:** CUDA (NVIDIA T4 GPU)
* **Data Engineering:** Data Augmentation (Random Rotations & Flips), Class Weighting
* **Evaluation Metrics:** Scikit-Learn (Classification Report, Confusion Matrix), Seaborn/Matplotlib

---

## ⚖️ Solving the Data Engineering Challenge: Class Imbalance
The initial dataset contained a severe **3:1 class imbalance** (3,875 Pneumonia images vs. 1,341 Normal images). 

1. **The Problem:** In the baseline training run, the model achieved an overall accuracy of 78% but suffered from a low healthy lung recall of **41%** because it biased its predictions toward the majority class (Pneumonia).
2. **The Solution:** Instead of accepting flawed results, **Class Weights** were calculated and passed directly into the loss function penalty structure. By punishing the model roughly 2.89x harder for missing a "Normal" image, the network balanced its predictions perfectly—skyrocketing the overall accuracy to **88%**.




## 🌐 Live Web Demo
You can try the model yourself here: [https://huggingface.co/spaces/dibyaswayam/pneumonia-ai-detector](https://huggingface.co/spaces/dibyaswayam/pneumonia-ai-detector)
