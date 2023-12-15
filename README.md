# Custom Object Detection Model Training with Ultralytics YOLOv8 
### Introduction

This tutorial provides a comprehensive guide on training a custom object detection model using YOLOv8 from Ultralytics. It covers the complete process including setting up the environment in Google Colab, labeling data, exporting it to YOLOv8 format, and conducting the training process.

### Prerequisites

- Google Colab account.
- Custom dataset for object detection.
- Basic understanding of Python and object detection concepts.

### Setup and Installation

1. **Check GPU Availability**: Ensure you're using a GPU in Google Colab for efficient training.
2. **Install Ultralytics YOLOv8**: Install the necessary YOLOv8 package from Ultralytics for model training.

### Data Preparation

- **Labeling and Exporting Dataset**: Use tools like Roboflow for dataset labeling and export it in YOLOv8 format.
- **Importing Dataset into Colab**: Download your prepared dataset into the Google Colab environment.

### Model Training

- **Import YOLO Model**: Import the YOLO module from Ultralytics.
- **Load Dataset**: Make sure the dataset paths are correctly set in Colab.
- **Customize Training Parameters**: Set model type, epochs, and image size according to your requirements.

### Monitoring and Evaluation

- **Track Training Progress**: Monitor metrics such as mean average precision (mAP) and observe validation graphs.
- **Model Validation and Prediction**: Validate the trained model on new data and use it for making predictions.

### Exporting and Utilizing the Trained Model

- **Export Trained Weights**: Instructions on exporting the trained model weights for further usage.
- **Integration into Custom Scripts**: Guidelines on integrating the trained model into custom Python scripts for practical applications.

### Conclusion

The tutorial provides a step-by-step approach to training a custom object detection model using YOLOv8 in Google Colab, preparing you for subsequent steps like live inference with the trained model.


# Training YOLOv8 with Custom Dataset in Google Colab
## Overview

This guide covers the process of training a custom object detection model using Ultralytics YOLOv8 in Google Colab.
This tutorial guides you through training a custom object detection model using YOLOv8 from Ultralytics. We'll cover setting up your environment, labeling data with various tools, exporting data to YOLOv8 format, and importing it into Google Colab. The goal is to train a custom object detection model tailored to your needs.

## Prerequisites

- Google Colab account
- NVIDIA GPU access in Colab
- Roboflow account

## Steps

1. **Environment Setup**:
   - Set UTF-8 encoding.
   - Check the available GPU: Ensure that you're running on a GPU by setting the runtime type to GPU in Google Colab.
    ```python
    # Set UTF-8 encoding.
    import locale
    locale.getpreferredencoding = lambda: "UTF-8"
    # Check NVIDIA GPU status
    !nvidia-smi
    ```
2. **Install Required Libraries**:
   - Install Ultralytics YOLO.
   - Install Roboflow.
    ```python
    # Install Ultralytics YOLO package
    !pip install ultralytics  
    # Install Roboflow package
    !pip install roboflow
    ```
3. **import packages and YOLO**
    ```python
    # import packages and the YOLO package
    from ultralytics import YOLO
    import os
    from IPython.display import display, Image
    from IPython import display
    # Clear the output to tidy up notebook
    display.clear_output()
    ```
4. **Data Preparation**:
   - Download your dataset from Roboflow.
    ```python
    from roboflow import Roboflow
    # Replace with your actual API key
    rf = Roboflow(api_key="your_api_key")  
    project = rf.workspace("your_workspace").project("your_project")
    # Download dataset
    dataset = project.version(1).download("yolov8")  
    ```
5. **Training the Model**:
   - Train the model using different optimizers (SGD and AdamW).
   - Training the Model (SGD Optimizer):
   ```python
   # Train model with SGD optimizer
   !yolo task=detect mode=train model=yolov8m.pt data={dataset.location}/data.yaml epochs=200 imgsz=640 patience=100 optimizer='SGD'  
   ```
   - Training the Model (AdamW Optimizer):
   ```python
   # Train model with AdamW optimizer
   !yolo task=detect mode=train model=yolov8m.pt data={dataset.location}/data.yaml epochs=200 imgsz=640 optimizer='AdamW' lr0=0.001 patience=100  
   ```
6. **Model Evaluation**:
   - Display the confusion matrix and training results for model evaluation.
    ```python
    # Display confusion matrix
    Image(filename=f'your path/confusion_matrix.png', width=600)
    # Display training results
    Image(filename=f'your path/results.png', width=600)
    ```
7. **Model Validation**:
   - Validate the trained model on a separate dataset.
    ```python
    # Validate the model
    !yolo task=detect mode=val model=/content/runs/detect/train3/weights/best.pt data={dataset.location}/data.yaml
    ```
8. **Running Predictions**:
   - Perform object detection on new data using the trained model.
   ```python
   # Run predictions on a video file
   !yolo task=detect mode=predict model=/content/runs/detect/train3/weights/best.pt conf=0.9 source=/content/videoplayback.mp4
    ```
9. **Exporting Results**:
   - Zip the prediction results for download and further use.
   ```python
    !zip -r folder_name.zip /content/runs/detect/predict  # Zip the prediction results for easy download
    ```

## Note

Replace placeholder values like `your_api_key`, `your_workspace`, and `your_project` with your actual Roboflow API key and project details.

## Conclusion

This guide simplifies the process of training and evaluating a custom YOLOv8 model in Google Colab, enabling you to apply object detection to your specific use cases.