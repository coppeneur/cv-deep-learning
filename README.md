# cv-deep-learning

## Emotion Detection Demo

The `demo.py` script is designed to perform emotion detection on a **video file** or **live webcam** feed using a our chosen model "IntermediateCNN". 
The script automatically loads the best pre-trained model.

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
