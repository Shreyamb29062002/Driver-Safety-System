# Drowsiness Detection System

## Introduction

This script detects drowsiness, yawning, and face presence using a webcam. It employs computer vision techniques using OpenCV and dlib to analyze facial landmarks and plays alarm sounds to alert the user.

## Prerequisites

Ensure you have the following Python packages installed:

- `numpy`
- `opencv-python`
- `dlib`
- `scipy`
- `imutils`
- `playsound`

You can install these packages using pip:
```bash
pip install numpy opencv-python dlib scipy imutils playsound

Additionally, you need the pre-trained shape predictor model from dlib: shape_predictor_68_face_landmarks.dat.

Script Overview
Functions
trigger_alarm: Plays an alarm sound for general alerts.
trigger_voice: Plays a voice alert when no face is detected.
trigger_yawning: Plays an alert when yawning is detected.
trigger_drowsiness: Plays an alert when prolonged eye closure is detected.
cal_yawn: Calculates the distance between the upper and lower lips to detect yawning.
cal_ear: Calculates the Eye Aspect Ratio (EAR) to detect eye closure.
Main Logic
Camera Setup: Opens a connection to the webcam.
Models Initialization: Loads face detector and facial landmarks predictor.
Thresholds: Sets thresholds for yawning and eye aspect ratio.
Eye Closure Tracking: Initializes variables to track the duration of eye closure.
Detection Loop
Frame Capture: Reads a frame from the webcam.
FPS Calculation: Calculates the frames per second for performance monitoring.
Face Detection: Detects faces in the grayscale frame.
No Face Detected: Triggers a voice alert if no face is detected.
Landmarks Detection: Detects facial landmarks if a face is found.
Yawn Detection: Checks the distance between the lips to detect yawning.
Eye Closure Detection: Checks the Eye Aspect Ratio (EAR) to detect eye closure and possible drowsiness.
Alerts: Plays appropriate alerts based on the detected conditions (yawning, drowsiness).
Termination
The script runs indefinitely until the 'q' key is pressed to quit.
Releases the webcam and destroys all OpenCV windows on exit.