# DEMOS
- different Demos to show use cases of Ultrasound imaging in HCI
- demos are build to work with provided Data
- detection parameters can easily be adapted to fit different videos (TODO)

## Setup
1. Run the demos with the provided videos.
   1. Nothing to set up. See [##Usage]
2. Run the demos with your own ultrasound videos
   1. Record a mp4 and drop it into the data folder
   2. See [##Usage]
3. Use a live ultrasound video feed
   1. See the documentatio of [##My_Setup]
   2. Good Luck

### Setup Documentation
<!-- Explain how detection works (what US machine, capture device, settings, capture software, ...) -->
Required soft/hardware:
- Ultrasound machine: 
  - Philips Sono Diagnost R-1200 (SDR-1200)
  - Probe LA 3510 (3.5 MHz)
  - Lateral Resolution: 1.6 mm (for LA 3510)
  - Full view field: 102 x 180 mm
  - FPS: between 24.4 and 5.1 (depends on zoom factor and focus settings)
  - 32 Graustufen
- Capture Device:
  - Easy Cap (Analoge to Digital)
  - Windows 11 Driver
- Capture software
  - OBS Studio

Workflow:
- Get the video output of the ultrasound machine with a capture device
- Capture with OBS
  - Set the video to PAL_B
  - Start a virtual camera
- Start the demos in python
  - Might have to change the `VIDEO_ID` in `Player.py` to match the virtual camera

## Usage
- (run `py main.py -d 1` (number indicate the demo you want to run))
- run from each Demo separately 
  - run `py demo-name.py` to run the demo using an existing demo-video or using a video stream (might have to change video id (in code))
  - run `py demo-name-py -src path/to/video.mp4` to run the demo using custom videos
    - or change the paths in the files

## Overview
Goals:
- create demos that show off different abilities of ultrasound imaging
- create demos to extend the usage of ultrasound imaging into HCI
<!-- Top-Level explanation of Demos (why, what, how) -->
<!-- "Nav" to the detailed README's + short summary + image -->
<!-- Explain choices of detection method(s): why, how, expansion options, ... -->

### 1. 3D Scanner
demo for: acoustic properties of different materials
- detect an object and recreate its cross section in 3D
- move the object to gather and append the cross sections
- use a reference point to estimate the position
- TODO

### 2. Painter
demo for: depth values
- `painter.py`
- maybe use a "pen" instead of a finger for more consistent detection?
- thresholds can change depending on footage -> sometimes a higher/lower value gets better detection
- [ ] add a line-stop when the finger is remove for long enough (more than one frame at least)

### 3. Deformable Interaction
demo for: tangible interaction (HCI application)
- `deformer.py`
- using a deformable stressball to create input interactions
- map movement to "rolling the ball around"
- map two different actions (e.g. jump, crouch) to squish (compress) and squash (flatten)

### 4. Water Flow
demo for: noise of conductive material and Gestures (HCI application)
- `gesturer.py`
- use the bubbles "visible" in water to ...
  - recognize gestures (e.g. circle = select) (see also: 1$ recognizer)
  - (create flowing, moving paintings)
  - ...?

### 5. Physics Visualization/Simulation ?
demo for: reflections of sound waves
- `visualizer.py`
- simulate the sound waves interacting with different materials
- or create a game based on reflection, e.g. Pong (have to reflect ball)
- ...?

### Ideen
- USI: light robustness, reflections, (depth values), (acoustic properties), speckle, see through materials
- HCI: Tangibles, Gestures, 

- reflections:
  - Sound from multi-reflections (-> e.g. moving plastic sheet causes many lines -> make (electronic) cool sound effect)
  - Visualization of sound waves and reflections
  - Pong with different "reflection" speeds/strength according to how strong materials shows up
  - something with the mirrors
- speckle: 
  - [ ] use bubbles to detect water movement (prob. with optical flow)
    - make swirly paintings to project into a dark room
    - map to actions: move a character or your mouse across the screen
    - Soundification
  - aka specktical flow, lol
- See through:
  - Guessing game: draw something behind opaque wall other has to guess (demo=shows solution) (not really a demo)
