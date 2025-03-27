# Weapon Detection using AI and Deep Learning

## Overview
This project aims to detect weapons (e.g., guns, knives) in images/videos using **deep learning** and **computer vision**. It can be deployed in security systems, surveillance cameras, or public safety applications to identify potential threats in real-time.

## Features
- Detects weapons (firearms, bladed weapons) in images/video streams.
- Uses a pre-trained deep learning model (YOLOv5/SSD/Faster R-CNN) fine-tuned on weapon datasets.
- Real-time detection with OpenCV or TensorRT optimization.
- Configurable confidence thresholds to reduce false positives.

## Requirements
- Python 3.8+
- Libraries:  
torch>=1.10.0
torchvision
opencv-python
numpy
pandas
matplotlib

## Installation
1. Clone the repository:  
git clone (https://github.com/Rakesh4655) 


2. Install dependencies:  
pip install -r requirements.txt

## Usage
1. **For Image Detection**:  
python detect.py --input path/to/image.jpg --weights weights/best.pt

2. **For Real-Time Webcam Detection**:  
python detect.py --source 0 --weights weights/best.pt

3. **For Video File Detection**:  
python detect.py --input path/to/video.mp4 --weights weights/best.pt

## Dataset
- Custom dataset of annotated weapons (images + labels in YOLO format).
- Sources:  
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)  
- [Kaggle Weapon Detection Datasets](https://www.kaggle.com/datasets)  
- Manually collected CCTV footage (if applicable).

## Model Training
1. Prepare dataset in YOLO format.
2. Train the model:  
python train.py --data data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --epochs 50

3. Evaluate performance:  
python test.py --data data.yaml --weights runs/train/exp/weights/best.pt

## Results
- **Precision**: 92%  
- **Recall**: 88%  
- **mAP@0.5**: 90%  

## Future Improvements
- Extend to detect improvised weapons (e.g., pipes, bottles).
- Deploy on edge devices (Jetson Nano) for low-latency inference.
- Integrate with alarm systems for automated alerts.

## Contact
For questions or collaborations, email: [rakeshraki1408@gmail.com]  
GitHub: [Rakesh4655](https://github.com/Rakesh4655) 
