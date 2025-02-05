# **YOLOv9-Based Weed Detection for Sustainable Agriculture**

## **Overview**
This repository contains the implementation of a **semi-supervised deep learning model** for weed detection in precision agriculture using **YOLOv9**. The model leverages **semi-supervised learning** to improve detection accuracy while reducing the dependency on labeled data. It utilizes **Generalized Intersection over Union (GIoU) loss** and **F1 score optimization** for enhanced object detection.

This work is part of a **competition submission**, demonstrating advanced deep learning techniques for **real-time weed detection** in agricultural settings.

---

## **Project Structure**
- `included_aug_mcc1.ipynb` – Data augmentation strategies and model pre-processing.
- `v9-giou-f1.ipynb` – Model training and evaluation using **GIoU loss** and **F1 score-based optimization**.
- `Template_for_Geophysical_Journal_International_GJIRAS.pdf` – Research paper detailing the methodology and results.
- `README.md` – Documentation for the project.

---

## **Requirements**
To run this project, install the following dependencies:

```bash
pip install torch torchvision albumentations numpy opencv-python matplotlib tqdm ultralytics
```

**Key Libraries Used:**
- `PyTorch` – Deep learning framework for training the YOLOv9 model.
- `Albumentations` – Advanced data augmentation library.
- `OpenCV` – Image processing and transformation.
- `NumPy` – Numerical computations.
- `Matplotlib` – Visualization of predictions and model performance.

---

## **Dataset & Preprocessing**
The dataset consists of:
- **200 labeled images** (annotated with crops and weeds).
- **1,000 unlabeled images** (used for semi-supervised training).

### **Data Augmentation**
To enhance the model’s robustness, data augmentation was applied using **Albumentations**:
- **RandomBrightnessContrast**
- **HueSaturationValue**
- **RandomFog, RandomRain, RandomSnow**
- **MotionBlur, GaussianBlur, GaussianNoise**
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- **GridDistortion**

Each image was augmented **5 times** to simulate real-world field conditions.

---

## **Model Architecture: YOLOv9**
We utilize **YOLOv9**, an advanced version of YOLO (You Only Look Once), for real-time object detection.

### **Key Improvements in YOLOv9:**
- **Better localization accuracy** using **GIoU loss**.
- **Improved feature extraction** for small object detection.
- **Optimized training with F1-score-based loss function.**

### **Semi-Supervised Learning Approach**
1. **Initial Training**: The YOLOv9 model is trained using **200 labeled images**.
2. **Pseudo-Labeling**: The trained model predicts labels for **1,000 unlabeled images**.
3. **Confidence-Based Selection**: The top **200 high-confidence pseudo-labeled images** (confidence ≥ 0.5) are added to the dataset.
4. **Iterative Training**: The model is retrained using the expanded dataset.

---

## **Loss Functions & Optimization**
### **1. F1 Score Loss**
Used for optimizing the trade-off between **precision** and **recall**:

```
F1 = 2 * (Precision * Recall) / (Precision + Recall + 1e-8)
```

This ensures that both false positives and false negatives are minimized.

### **2. Generalized Intersection over Union (GIoU) Loss**
An improved IoU-based metric for better bounding box alignment:

```
L_GIoU = 1 - (IoU - |C - (A ∪ B)| / |C|)
```

Where:
- `A` = predicted bounding box
- `B` = ground truth bounding box
- `C` = smallest enclosing convex shape

Using **GIoU loss** helps the model better localize weeds under complex environmental conditions.

---

## **Confidence Thresholding**
The model's predictions are filtered using a **confidence threshold**:

```
Confidence = Objectness * Class Probability
```

Only predictions with a confidence score ≥ **0.5** are considered for pseudo-labeling.

---

## **Results & Performance Metrics**
### **Key Evaluation Metrics**
| Metric       | Performance |
|-------------|------------|
| **Precision** | 0.89 |
| **Recall**   | 0.88 |
| **F1 Score** | 0.89 |
| **mAP@50-95** | 0.62 |

### **Key Findings**
- The **semi-supervised YOLOv9 model** achieved high accuracy in weed detection.
- Using **F1 loss** improved balance between precision and recall.
- **GIoU loss** enhanced bounding box alignment.
- **Data augmentation** significantly boosted model robustness in real-world scenarios.

---

## **Conclusion**
This work demonstrates how **deep learning and semi-supervised learning** can revolutionize **weed detection in precision agriculture**. By integrating **YOLOv9** with **GIoU loss** and **F1 optimization**, we achieve **scalable, high-accuracy detection** while reducing reliance on **expensive labeled data**.

### **Future Improvements**
- Fine-tuning model hyperparameters for **real-time deployment**.
- Expanding the dataset to include **more diverse agricultural conditions**.
- Implementing **edge AI** for low-power **on-field inference**.

---

## **How to Run**
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/weed-detection-yolov9.git
cd weed-detection-yolov9
```

### **2. Run the Model Training**
Open and execute the Jupyter Notebook:
```bash
jupyter notebook v9-giou-f1.ipynb
```

---

## **Acknowledgments**
This project is submitted for **[Competition Name]**, using cutting-edge **YOLOv9 object detection** for **precision agriculture**.
