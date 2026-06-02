**Pixel Segmentation using YOLO for Flukeprint Detection in Whale Monitoring**

## Project Overview
This semester project, conducted at the **ECEO Laboratory**, focuses on automating the detection of whale "flukeprints" using thermal drone imagery. 

A flukeprint is the thermal signature left on the sea surface by the movement of colder water caused by a whale's body and tail strikes. The goal is to train a YOLO-based model to identify and segment these signatures within a labeled dataset, enabling automated monitoring of whale activity.

## Repository Structure

```text
.
├── F_Binary_Dataset/           # Binary dataset containing labeled thermal images
├── Behavioral_Dataset          # Behavioral and temporal class distinctions labeled thermal images
├── runs/                       # Training outputs and logs
├── 1_Exploratory_Phase.ipynb   # Notebook for testing YOLOv8n-seg parameters
├── 2_Final_Model.ipynb         # Comparative analysis (YOLOv8n vs YOLOv26n) & RayTune fine-tuning
├── 3_Behavior_Training         # Main results: fine-tuned binary model on the behavioral dataset through a phased training strategy (M0-M2).
├── *.pt                        # Pre-trained and custom model weights (yolov8n-seg.pt, yolo26n.pt, etc.)
└── README.md