# DEMOS
- different Demos to show use cases of Ultrasound imaging in HCI
- demos are build to work with provided Data
- detection parameters can easily be adapted to fit different videos (TODO)

## Setup
<!-- Explain how detection works (what US machine, capture device, settings, capture software, ...) -->

## Usage
- (run `py main.py -d 1` (number indicate the demo you want to run))
- run from each Demo separately 
  - run `py demo-name.py` to run the demo using an existing demo-video or using a video stream (might have to change video id (in code))
  - run `py demo-name-py -src path/to/video.mp4` to run the demo using custom videos

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
- maybe use a "pen" instead of a finger for more consistent detection?
- thresholds can change depending on footage -> sometimes a higher/lower value gets better detection
- [ ] add a line-stop when the finger is remove for long enough (more than one frame at least)

### 3. Deformable Interaction
demo for: tangible interaction (HCI application)
- using a deformable stressball to create input interactions
- map movement to "rolling the ball around"
- map two different actions (e.g. jump, crouch) to squish (compress) and squash (flatten)

### 4. Water Flow
demo for: noise of conductive material and Gestures (HCI application)
- use the bubbles "visible" in water to ...
  - recognize gestures (e.g. circle = select) (see also: 1$ recognizer)
  - (create flowing, moving paintings)
  - ...?

### 5. Physics Visualization/Simulation ?
demo for: reflections of sound waves
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
