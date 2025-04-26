# Meow Kitty

A Python project that processes video and detects cats.

## Features

* Opens and processes video files
* Displays the video in a window while processing
* Identifies when a cat is present in the video
* Prints "Meow kitty!" when a cat is detected
* Prints "Searching for kitty..." when no cat is detected
* Draws bounding boxes around detected cats

## Requirements

- Python 3.12+
- pipenv

## Installation

1. Clone this repository
2. Install dependencies using pipenv:

```bash
pipenv sync
```

This will install all required packages:
- opencv-python
- ultralytics

## Usage

Run the script with a path to a video file:

```bash
pipenv run python main.py /path/to/your/video.mp4
```

The script will:
1. Open a window displaying the video
2. Process the video frame by frame in sync with the display
3. Print and display:
   - "Meow kitty!" when a cat is detected
   - "Searching for kitty..." when no cat is detected
4. Draw green bounding boxes around detected cats
5. Print "Video processing complete!" when the video ends

### Controls
- Press 'q' at any time to quit the application

The video playback is synchronized with the processing, so you can watch the detection in real-time.

## How It Works

The project uses:
- OpenCV for video processing
- YOLOv8 (via ultralytics) for cat detection
- The COCO dataset's class 15 represents cats in the detection model

## Notes

- The first time you run the script, it will download the YOLOv8 model, which may take some time depending on your internet connection.
- For optimal performance, consider using a GPU-enabled system.
