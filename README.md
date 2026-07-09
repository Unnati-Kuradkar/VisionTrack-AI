VisionTrack AI

Author: Unnati Kuradkar
Affiliation: MCA Student
Date: July 2026

Abstract

VisionTrack AI is a real-time object detection and tracking system that utilizes the RF-DETR Nano model to identify and classify objects from a live webcam feed. The application provides instant visual feedback through bounding boxes, object labels, and live object counting. By combining advanced computer vision techniques with an interactive Streamlit dashboard, the system enables efficient monitoring and analysis of objects in real time.

Introduction

Computer Vision has become a key area of Artificial Intelligence, enabling machines to understand and interpret visual information from the world. Real-time object detection is widely used in surveillance, smart monitoring, autonomous systems, retail analytics, and industrial automation.

The VisionTrack AI project was developed to demonstrate the practical implementation of modern object detection models using live video streams. The system processes webcam footage in real time, detects multiple objects simultaneously, and displays detection results through an intuitive user interface.

Literature Review

Traditional object detection algorithms such as Haar Cascades and HOG-based detectors provided basic object recognition capabilities but struggled with accuracy and scalability.

Modern deep learning-based models including:

R-CNN
Fast R-CNN
Faster R-CNN
YOLO (You Only Look Once)
DETR (Detection Transformer)

have significantly improved object detection performance.

RF-DETR (Real-Time Fine-tuned Detection Transformer) combines transformer-based architectures with optimized detection pipelines, offering high accuracy while maintaining real-time performance.

Methodology
1. Video Capture

The system captures live video frames from the user's webcam using WebRTC.

2. Frame Processing

Each frame is resized and optimized for efficient inference.

3. Object Detection

The RF-DETR Nano model analyzes incoming frames and identifies objects present in the scene.

4. Object Classification

Detected objects are assigned class labels such as:

Person
Clock
Chair
Laptop
Bottle
Cell Phone

and many more supported classes.

5. Visualization

Detected objects are highlighted using:

Bounding Boxes
Class Labels
Live Object Count Overlay
6. Dashboard Display

Results are displayed through an interactive Streamlit dashboard for real-time monitoring.

Implementation
Programming Language
Python
Frameworks & Libraries
Streamlit
Streamlit-WebRTC
OpenCV
RF-DETR
Supervision
AV
Tools Used
Visual Studio Code
GitHub
Streamlit
Results and Discussion
Achievements

✅ Real-time webcam object detection

✅ Multiple object recognition

✅ Bounding box visualization

✅ Class label display

✅ Live object counting overlay

✅ Interactive dashboard interface

Sample Detection Output

The system successfully detects objects such as:

Person
Clock
Chair
Laptop
Mobile Phone

with real-time visual feedback.

Screenshots / Demo
Live Webcam Feed
Object Detection Dashboard
Real-Time Detection Results
Object Count Overlay
Limitations
Detection accuracy depends on lighting conditions.
Small or partially visible objects may not be detected accurately.
Performance varies depending on system hardware.
RF-DETR inference may be slower on low-end devices without GPU support.
Future Scope
Enhanced Tracking

Integrate advanced object tracking algorithms such as ByteTrack or DeepSORT.

Analytics Dashboard

Generate object detection reports and statistics.

Multi-Camera Support

Monitor multiple video streams simultaneously.

Cloud Deployment

Deploy the application on cloud platforms for remote access.

Alert System

Implement real-time notifications when specific objects are detected.

Custom Object Detection

Train the model on custom datasets for domain-specific applications.

Conclusion

VisionTrack AI demonstrates the practical application of Artificial Intelligence and Computer Vision in real-time object detection. By leveraging the RF-DETR Nano model, Streamlit, and OpenCV, the system provides accurate object recognition and visualization through an easy-to-use dashboard. The project showcases the potential of modern AI technologies for intelligent monitoring and real-world automation solutions.

References
RF-DETR Documentation – https://github.com/roboflow/rf-detr
OpenCV Documentation – https://opencv.org/
Streamlit Documentation – https://streamlit.io/
Supervision Documentation – https://supervision.roboflow.com/
Carion et al., "End-to-End Object Detection with Transformers (DETR)," Facebook AI Research, 2020.
Roboflow Computer Vision Resources – https://roboflow.com/
