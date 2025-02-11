# cv-deep-learning

## Overview

This repository contains the code for Assignment 2 of the "Computer Vision" course. 
Our primary pipeline is the `emotion_detection.ipynb` notebook, which includes the code for training and evaluating the emotion detection model. 
The src folder contains the main logic and code for the pipeline, organized into different modules for better clarity. 
Additionally, the `vit_transformer.ipynb` notebook contains the code for testing the ViT model on the FER2013 dataset.


## Emotion Detection Demo

The `demo.py` script is designed to perform emotion detection on either a video file or a live webcam feed using our chosen model, "IntermediateCNN."
It plots the detected emotions after processing each frame of the video or webcam feed.

### Prerequisites

Ensure you have the necessary dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```

### Usage

To run the `demo.py` script, you can use the following command in your terminal:

```bash
python demo.py [--video <path_to_video>]
```

### Arguments

- `--video` (optional): Path to a video file. If not provided, the script will use the webcam for real-time emotion detection.
- `-h` or `--help`: Displays the help message.

### Examples

#### Using a Video File

To run the demo on our test video file, use the following command:

```bash
python demo.py --video surprised-happy.mp4
```

#### Using Webcam

To run the demo using the webcam, simply omit the `--video` argument:

```bash
python demo.py
```
