MAIN2 - CNN v2 PHÂN LOẠI CHÓ/MÈO
============================================================

Project:
/content/drive/MyDrive/HOCSAU

Dataset:
/content/drive/MyDrive/HOCSAU/data/dataset_combined_main

Output folder:
/content/drive/MyDrive/HOCSAU/output/main2

MODEL FILES
------------------------------------------------------------
Best model:
/content/drive/MyDrive/HOCSAU/output/main2/best_cnn_cats_dogs_main2.keras

Latest model:
/content/drive/MyDrive/HOCSAU/output/main2/latest_cnn_cats_dogs_main2.keras

Final model:
/content/drive/MyDrive/HOCSAU/output/main2/final_cnn_cats_dogs_main2.keras

APP / CONFIG FILES
------------------------------------------------------------
App model info:
/content/drive/MyDrive/HOCSAU/output/main2/app_model_info.json

Class names:
/content/drive/MyDrive/HOCSAU/output/main2/class_names.json

Recommended threshold:
/content/drive/MyDrive/HOCSAU/output/main2/recommended_threshold.json

Model config:
/content/drive/MyDrive/HOCSAU/output/main2/model_config.json

Training info:
/content/drive/MyDrive/HOCSAU/output/main2/training_info.json

METRICS / REPORT FILES
------------------------------------------------------------
Training history:
/content/drive/MyDrive/HOCSAU/output/main2/history.csv

Test metrics:
/content/drive/MyDrive/HOCSAU/output/main2/test_metrics.json

Test predictions:
/content/drive/MyDrive/HOCSAU/output/main2/test_predictions.csv

Confusion matrix:
/content/drive/MyDrive/HOCSAU/output/main2/confusion_matrix.npy
/content/drive/MyDrive/HOCSAU/output/main2/confusion_matrix.png

Classification report:
/content/drive/MyDrive/HOCSAU/output/main2/classification_report.csv
/content/drive/MyDrive/HOCSAU/output/main2/classification_report.txt

PLOTS / EDA
------------------------------------------------------------
/content/drive/MyDrive/HOCSAU/output/main2/eda_class_distribution.png
/content/drive/MyDrive/HOCSAU/output/main2/eda_sample_images.png
/content/drive/MyDrive/HOCSAU/output/main2/accuracy_curve.png
/content/drive/MyDrive/HOCSAU/output/main2/loss_curve.png
/content/drive/MyDrive/HOCSAU/output/main2/correct_samples.png
/content/drive/MyDrive/HOCSAU/output/main2/wrong_samples.png

APP PREPROCESS NOTE
------------------------------------------------------------
- App nên mở ảnh bằng PIL, convert('RGB'), resize về (150, 150).
- Không chia pixel cho 255 trong app vì model đã có lớp Rescaling(1./255).
- Sigmoid output là xác suất dog.
- Nếu prob_dog >= recommended_threshold thì dự đoán Chó, ngược lại Mèo.