import argparse
import os
import time

import cv2
from ultralytics import YOLO


def process_video(video_path):
    """
    Process a video file to detect cats and print appropriate messages.
    Display the video in sync with processing.

    Args:
        video_path (str): Path to the video file
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    model = YOLO("yolov8n.pt")  # Using the nano version of YOLOv8 for efficiency
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / fps)  # Delay between frames in milliseconds

    # Create a window to display the video
    cv2.namedWindow('Cat Detection', cv2.WINDOW_NORMAL)

    print("Starting video playback. Press 'q' to quit.")

    # Process the video frame by frame
    while cap.isOpened():
        # Record the start time for frame rate control
        start_time = time.time()

        # Read a frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            break

        results = model(frame)

        # Check if a cat is detected
        cat_detected = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0].item())
                # Class 15 in COCO dataset is 'cat'
                if cls == 15:
                    cat_detected = True
                    # Draw bounding box for the cat
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 'Cat', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if cat_detected:
            print("Meow kitty!")
            cv2.putText(frame, 'Meow kitty!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print("Searching for kitty...")
            cv2.putText(frame, 'Searching for kitty...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Cat Detection', frame)

        # Calculate how long to wait to maintain the original video's frame rate
        processing_time = time.time() - start_time
        wait_time = max(1, int(frame_delay - (processing_time * 1000)))

        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Video processing complete!")


def main():
    parser = argparse.ArgumentParser(description='Detect cats in a video file.')
    parser.add_argument('video_path', type=str, help='Path to the video file')

    args = parser.parse_args()

    process_video(args.video_path)


if __name__ == "__main__":
    main()
