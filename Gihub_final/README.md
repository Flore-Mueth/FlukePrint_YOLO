# FlukePrint_YOLO
Pixel Segmentation using YOLO for Fluke Print detection in whale monitoring

Exploratory phase - testing yolov8n-seg differents parameters settings 
Final Model - yolov8n-seg vs yolo26n-seg AND RayTune final fine tuned param
Behav results trainonig ...



**Pixel Segmentation using YOLO for Flukeprint Detection in Whale Monitoring**

## Project Overview
This semester project, conducted at the **ECEO Laboratory**, focuses on automating the detection of whale "flukeprints" using thermal drone imagery. 

A flukeprint is the thermal signature left on the sea surface by the movement of colder water caused by a whale's body and tail strikes. The goal is to train a YOLO-based model to identify and segment these signatures within a large labeled dataset, enabling automated monitoring of whale activity.

## 📂 Repository Structure

```text
.
├── F_Binary_Dataset/       # Dataset containing labeled thermal images
├── runs/                   # Training outputs and logs
├── Exploratory_Phase.ipynb # Notebook for testing YOLOv8n-seg parameters
├── Final_Model.ipynb       # Comparative analysis (YOLOv8n vs YOLOv26n) & RayTune fine-tuning
├── *.pt                    # Pre-trained and custom model weights (yolov8n-seg.pt, yolo26n.pt, etc.)
└── README.md