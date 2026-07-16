# DonkeyCar Autonomous Vehicle

A Raspberry Pi–based autonomous RC car that uses computer vision to follow a marked lane and respond to visual road markers.

This project was developed as a hands-on introduction to autonomous vehicle software, integrating camera perception, image processing, steering control, and real-time vehicle actuation through the DonkeyCar platform.

## Project Overview

The vehicle processes images from its onboard camera to:

* Detect and follow a yellow lane marker
* Calculate steering commands from the lane position
* Smooth steering changes between camera frames
* Adjust throttle based on the severity of a turn
* Detect a red stop marker
* Stop the vehicle temporarily when the marker is detected
* Display steering and throttle telemetry on the camera output

The current implementation uses classical computer vision rather than a trained neural network.

## Current Features

### Yellow-Lane Detection

The camera image is converted from RGB or BGR into HSV color space. A configurable yellow-color range is then used to create a binary mask of the lane.

The controller analyzes the lower portion of the camera image because this region contains the lane immediately in front of the vehicle.

### Steering Control

The horizontal position of the detected lane is compared with the center of the camera frame.

The resulting error is normalized into a steering command between `-1.0` and `1.0`:

* Negative values steer in one direction
* Positive values steer in the opposite direction
* Values near zero keep the vehicle moving straight

Previous steering values are used to smooth sudden changes and reduce oscillation.

### Adaptive Throttle

The vehicle reduces its throttle during sharper turns and drives faster when the detected lane is close to the center of the frame.

This improves stability and reduces the chance of the vehicle leaving the lane.

### Red Stop-Marker Detection

The software creates two HSV masks to capture both ends of the red hue range.

When the number of detected red pixels exceeds a configured threshold, the controller sets steering and throttle to zero and stops the vehicle for approximately five seconds.

## Repository Structure

```text
Donkeycar/
├── lane_follow.py    # Basic yellow-lane following controller
├── mylane.py         # Lane following with red-marker detection and telemetry
├── myline.py         # Contour-based lane following and red-marker detection
└── README.md
```

### `lane_follow.py`

A basic lane-following implementation that:

* Crops the lower 40% of the image
* Detects yellow pixels in HSV color space
* Calculates the average horizontal lane position
* Generates a smoothed steering command
* Stops when too few lane pixels are detected

### `mylane.py`

An expanded controller that:

* Detects a red stop marker
* Uses image moments to calculate the yellow lane centroid
* Adjusts throttle based on steering magnitude
* Adds steering and throttle telemetry to the output image
* Returns the processed image for visualization

### `myline.py`

A contour-based implementation that:

* Applies erosion and dilation to clean the lane mask
* Selects the largest detected yellow contour
* Calculates its centroid
* Smooths steering commands
* Adjusts throttle according to turn severity
* Stops temporarily when a red marker is detected

## Hardware

The intended system includes:

* DonkeyCar-compatible RC vehicle
* Raspberry Pi
* Raspberry Pi camera or compatible USB camera
* Steering servo
* Electronic speed controller
* DC drive motor
* Battery and vehicle chassis
* Track with a visible yellow lane marker
* Red visual marker for stop testing

Exact hardware and calibration values may vary between vehicle configurations.

## Software Requirements

* Python 3
* DonkeyCar
* OpenCV
* NumPy
* Raspberry Pi OS or another DonkeyCar-compatible environment

Install the core image-processing dependencies with:

```bash
python3 -m pip install opencv-python numpy
```

For installation of the complete DonkeyCar framework, follow the official DonkeyCar setup instructions for your Raspberry Pi and vehicle hardware.

## Using a Controller

The files in this repository define controller classes intended to be added as parts within a DonkeyCar vehicle pipeline.

Example:

```python
from myline import LineFollower

lane_controller = LineFollower()

vehicle.add(
    lane_controller,
    inputs=["cam/image_array"],
    outputs=[
        "pilot/angle",
        "pilot/throttle",
        "cam/image_array",
    ],
)
```

The exact input and output names should be adjusted to match the configuration of your DonkeyCar application.

Run the vehicle through the normal DonkeyCar command after integrating the selected controller:

```bash
python3 manage.py drive
```

## Processing Pipeline

```text
Camera frame
     ↓
Crop region of interest
     ↓
Convert image to HSV
     ↓
Create yellow and red masks
     ↓
Locate lane centroid or contour
     ↓
Calculate horizontal lane error
     ↓
Generate smoothed steering command
     ↓
Adjust throttle for turn severity
     ↓
Send commands to the vehicle
```

## Parameters to Tune

The controllers contain several values that should be calibrated for the specific track and lighting conditions.

### Yellow HSV Range

```python
lower_yellow = np.array([18, 80, 80])
upper_yellow = np.array([40, 255, 255])
```

Adjust these limits when the lane is not detected reliably.

### Red Detection Threshold

```python
if red_pixels > 500:
```

Increase the threshold if the vehicle stops because of unrelated red objects. Decrease it if the intended stop marker is not detected.

### Steering Smoothing

```python
steering = 0.7 * previous_steering + 0.3 * new_steering
```

More weight on the previous value produces smoother but slower steering. More weight on the new value creates faster but potentially less stable corrections.

### Throttle

Throttle values should initially be kept low during testing. Increase them only after lane detection and steering are reliable.

## Testing Procedure

1. Raise the vehicle so its wheels can rotate without touching the ground.
2. Verify that the camera feed is available.
3. Display the yellow mask and confirm that the lane is isolated.
4. Move the lane marker left and right in front of the camera.
5. Confirm that the steering direction is correct.
6. Reverse the steering sign if the vehicle turns away from the lane.
7. Test at low throttle on a clear track.
8. Tune the HSV thresholds under the track’s actual lighting.
9. Test the red marker at different distances.
10. Increase speed gradually after stable operation is achieved.

## Limitations

The current controller relies on fixed HSV color thresholds. Its performance may therefore change because of:

* Shadows
* Reflections
* Uneven lighting
* Camera exposure
* Lane colors similar to surrounding objects
* Motion blur
* Track intersections
* Missing or obstructed lane sections

Red-marker detection is based on the number of red pixels rather than object classification. Any sufficiently large red object may trigger a stop.

The software should be treated as an experimental prototype and tested in a controlled environment.

## Future Development

Planned improvements include:

* More robust lane estimation
* PID-based steering control
* Stop-sign object detection
* ArUco marker recognition
* Intersection handling
* Route selection using visual markers
* Camera calibration
* Obstacle detection
* Sensor fusion
* Recording perception and control telemetry
* Integration of learned DonkeyCar driving models
* Combined lane, stop-marker, and navigation behavior

## Skills Demonstrated

This project applies:

* Python
* OpenCV
* NumPy
* Embedded Linux
* Raspberry Pi development
* Computer vision
* Image segmentation
* Contour detection
* Feedback-based steering
* Vehicle control
* Real-time debugging
* Autonomous robotics

## Safety

Operate the vehicle only in a controlled area away from roads, people, animals, and fragile objects.

Always begin testing with:

* Low throttle
* Accessible emergency power control
* Clear space around the vehicle
* The wheels lifted during initial actuator testing

## Contributors

Developed as a collaborative autonomous-vehicle project using the DonkeyCar platform.

## License

No license has currently been specified for this repository. Add a license before distributing or allowing reuse of the project.
