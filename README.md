# **Semi-Supervised Weed Detection**

## **Overview**
This project focuses on semi-supervised weed detection using a combination of 200 labeled images and 1000 unlabeled images of crops and weeds. The goal is to effectively leverage the small labeled dataset and the large unlabeled dataset to train a weed detection model that performs well across various conditions.

## **Data Preprocessing and Augmentation**
We have employed the albumentations library for data augmentation, applying transformations such as RandomBrightnessContrast, HueSaturationValue, RandomFog, RandomRain, RandomSnow, MotionBlur, GaussianBlur, GaussianNoise, CLAHE, and GridDistortion. For each image, 5 augmented images are generated. These augmentations simulate various environmental conditions like changes in lighting, weather, and image noise, making the model more adaptable to real world scenarios.

## **Requirements**
Ensure the following dependencies are installed before running the notebooks:

```bash
pip install torch numpy pandas matplotlib opencv-python tensorflow glob2 pyyaml shutil ultralytics
```
Alternatively, the notebooks can be run in **Google Colab**, where all dependencies can be installed as required.

## **Loss Functions for Weed Detection**

   ## **F1 Loss**
   In our weed detection model, we have utilized the F1 Loss to ensure a balanced evaluation of performance, especially when false positives and false negatives hold equal importance. This approach is           particularly effective in real-world scenarios where class imbalance is common.The F1 loss is derived from precision and recall, with the F1 score being the harmonic mean of these two crucial metrics. To prevent numerical instability and enhance reliability, a small epsilon (1e-8) is added to the denominator.

  Precision measures the proportion of positive predictions that are actually correct.
  Recall (or sensitivity) determines the proportion of actual positives that are correctly identified by the model.

  By leveraging the harmonic mean, the F1 score mitigates the risk of misleading evaluations caused by imbalanced precision and recall. This ensures that the model performs well in both aspects to achieve a high   F1 score, making it a robust metric for assessing performance in weed detection.

 ## **GIoU Loss**

To further enhance our model’s accuracy, we have implemented Generalized Intersection over Union (GIoU) Loss. Traditional Intersection over Union (IoU) primarily focuses on the overlap between predicted and ground truth bounding boxes, but it does not account for spatial misalignment.

GIoU Loss addresses this limitation by penalizing bounding boxes that are misaligned with the ground truth. This improvement is particularly crucial in weed detection, where environmental complexities can make object localization challenging. By integrating GIoU loss into our model, we enable better object localization, ultimately enhancing both precision and recall.


Through the combination of F1 Loss and GIoU Loss, our weed detection model achieves superior performance, leading to more precise and reliable weed management for precision agriculture.
