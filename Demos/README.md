# DEMOS
- different Demos to show use cases of Ultrasound imaging in HCI
- demos are build to work with provided Data
- detection parameters can easily be adapted to fit different videos (TODO)

## Usage

### Setup

**1st Option**: run the demos with the provided videos. Requires no additional setup.

**2nd Option**: run the demos with your own ultrasound videos:
  - drop your ultrasound videos in the `Data`-folder
  - this will require updating some of the demo-specific parameters

**3rd Option**: run the demo in a live setting:
  - see [live usage](#live-usage) section
  - Good Luck!

### Run the Demos

Clone the repository, create a .venv and install the requirements inside it. Then navigate to the `Demos`-Folder
```sh
git clone https://github.com/no-ls/MT_USInteraction.git

py -m venv .venv
.venv/Scripts/activate

py -m pip install -r requirements.txt

cd Demos
```


From there you can **run the demos** using the provided videos or by using a live stream from a virtual camera
```sh
py demo-name.py
```
or run them using your **own** videos
```sh
py demo-name.py -src path/to/video.mp4
```

### Controls
All demos use the same basic structure for playing and interacting with the footage (see: `Helpers`-Folder). Application-specific controls are marked as such:

|Key|Function|Application|
|:-:|:-------|----------:|
|<kbd>s</kbd>|save a screenshot of the current running demo in the `Data/Out`-Folder|all|
|<kbd>d</kbd>|toggle debug information|all|
|<kbd>↓↑</kbd>|adjust the segmentation parameters (threshold or color quantization), or use the slider|all|
|<kbd>r</kbd>|reset the application|all but scanner.py|
|<kbd>Enter</kbd>|start/stop a 3D scan (only works when OpenCV window is focused)|scanner.py|
|<kbd>f</kbd>|Toggle freehand-scan mode (still requires start/stop-ing scan)|scanner.py|


### Live Usage

#### Required equipment
1. Ultrasound machine, with video output
2. Capture Device (Analoge to Digital) with driver (e.g. Easy Cap)
3. Capture software (e.g. OBS Studio) or similar
4. Acrylic tank filled with water
5. Ultrasound gel

I used the *Philips Sono Diagnost R-1200* (SDR-1200) with the *LA 3510* probe (3.5 MHz). It has a lateral resolution (aka how far apart objects can be placed next to each other) or 1.6mm. A full view field of: 102x180 mm for the x1 zoom setting. Mostly I used the default x1.5 setting with 102x138mm. The machine returned 24.4 fps (1 focus point), 10.4 fps (2 focus points) or 5.1 fps (3 focus points, I never used that many). You'd likely also want to only use 1 or 2 focus point to not degrade performance too much, specifically for optical flow only 1 focus point is optimal.

![Diagram of the pipeline to get the demos working live](../Data/Images/pipeline.png)

#### Workflow
- Fill the acrylic tank with water and place the ultrasound probe against the side of it (use ultrasound gel as a coupling agent)
- Get the video output of the ultrasound machine with a capture device
- Capture with OBS
  - Set the video to PAL_B
  - Start a virtual camera
- Start the demos in python
  - Might have to change the `VIDEO_ID` in `Player.py` to match the virtual camera. Currently set to 2 

## Demo Overview
<!-- Goals:
- create demos that show off different abilities of ultrasound imaging
- create demos to extend the usage of ultrasound imaging into HCI -->
<!-- Top-Level explanation of Demos (why, what, how) -->
<!-- "Nav" to the detailed README's + short summary + image -->
<!-- Explain choices of detection method(s): why, how, expansion options, ... -->

### 1. 3D Scanner
demo for: 3D reconstruction
- `scanner.py`

The application can be used freehand or by utilizing a scan-bed-construction, that calculates the current depth or height of the object with the help of a diagonal stick (pointing backwards) and triangulation.

![Example of the painter demo - scan-bed](../Data/Images/scanbed.jpg)

Contours are extracted by using color quantization and then parsed and stacked into a point cloud.
Due to the nature of ultrasound images, noise and artifacts can be difficult to fully avoid.
For good results (see images below) a ultrasound agar phantom can be used. 

Measurements I typically used: 
- 125g water + 6.3g agar agar 
- stir powder into cold water
- boil and pour into a mold

For further information and the original recipe see this [paper](https://doi.org/10.1016/j.afjem.2015.09.003) by Earle et al.

The images below show the segmentation and resulting point cloud of a 3D scan with this demo.

![Example of the scanner demo - segmentation](../Data/Images/repo-scanner.png)
![Example of the painter demo - point cloud](../Data/Images/repo-scanner2.png)



### 2. Painter
demo for: depth values
- `painter.py`

Use a finger to draw lines into the water bath. The application detects the contour and uses its center movement to draw lines. The size of the contour determines the line thickness. A line break (and color change) is caused when no significant contour can be detected (aka by removing your finger).

Uses a simple threshold to segment the contours. Adjust to your needs.

![Example of the painter demo in action](../Data/Images/repo-painter.png)

### 3. Deformable Interaction
- `deformer.py`

Demonstrates the use of a deformable material for interaction. Requires a stressball or similarly squishy material (tested with stress ball made out of TPE, anything else might require adjustments). Uses color quantization to segment the contour of the stressball from the background. Fits an ellipsis around the contour to get its major and minor axis, and use them to differentiate between squish and squash actions. They mimic right and left mouse clicks, respectively.
The center of the balls contour is used to simulate mouse movement.

![Example of the deformer demo in action](../Data/Images/repo-deformer.png)

### 4. Water Flow
- `flow.py`

Uses air bubbles introduced to the water from movement to track the motion of the water flow. These bubbles are segmented using an adaptive threshold and the processed frames are used for optical flow calculation. 
A ball is drawn over the frames, and the flow vectors are used to move it along the screen.

![Example of the flow demo in action](../Data/Images/repo-flow.png)


### 5. Reflection Visualization aka "Pong"
- `pong.py`

Visualizes the way a single ultrasound reflection works by creating a pong-style game. A ball (the reflection) moves down the screen. If it comes into contract with a strong enough reflector (e.g. your finger), its reflected. 
Score points by reflecting the ball back up the screen.

The segmentation is done using color quantization and uses the incident path, recreates the reflecting line (using the segmented contour), and then calculates the new reflection path.

![Example of the pong demo in action](../Data/Images/repo-pong.png)

