# NOTE
- quick build for testing, but not really used much
  - (only threshold and k-means are implemented, (grab-cut does not really work))
  - OpenCV has existing [sample](https://github.com/opencv/opencv/tree/4.x/samples/python) implementations for graph cut (aka grabcut), optical flow and watershed 
- can be expanded for proper comparison, but for my purposes not necessary
- can be used to quickly test different threshold values

## Usage
- install cv2 (e.g. with pip)
- run: `py main.py -src path/to/png/or/mp4/file`
- use number keys, e.g. <kbd>1</kbd> for threshold, to switch between algorithms
- use <kbd>arrow up</kbd> and <kbd>arrow down</kbd> to change the values used in the algorithm (e.g. threshold value)
- <kbd>q</kbd> or <kbd>esc</kbd> to quit
- <kbd>s</kbd> to save an image/frame


## Notes
before abandonment...

### Algorithms to Compare
- [x] thresholding (+ contour detection)
  - works without initialization
  - good fps ~20
- [x] k-means and/or fuzzy C-means
  - even with k=2 (you don't really need much more) very slow: fps ~6
  - gets contours pretty well
- [ ] graph cuts[^1] ([grab cut](https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html))
  - requires initialization (does not work yet), can probably be adjusted live
  - bad performance -> would prob. require GPU use with CUDA (make with [install](https://coderivers.org/blog/install-opencv-python-with-cuda/))
  - Test: Test/grabcut.py (von OpenCv)
- [ ] [optical flow](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html) ?
  - prob. good for application, where only the motion is important, not the detection -> e.g. Finger-Drawing
  - Test/opt_flow.py
- [ ] PDMs[^1] (deformable models) ?
- [ ] existing medical DL Model
  - e.g. U-Net
  - U-Net
    - pre-trained US model [here](https://github.com/izazahmad-ai/Ultrasound-Image-Segmentation-using-U-Net)
      - -> segments tumors and lesions -> objects would have to look like those
    - pre-trained model for high-definition images in PyTorch [here](https://github.com/milesial/Pytorch-UNet) 
      - trained on the Carvana dataset (Cars)
    - pre-trained models for lesion segmentation in US images [here](https://github.com/Yingping-LI/Light-U-net)
      - in folder `pretrained_models`
    - TODO: 3D ones + get actual images
- [ ] Meta AI [segment anything](https://segment-anything.com/)[^1] aka [SAM](https://github.com/facebookresearch/segment-anything) or [SAM2](https://github.com/facebookresearch/sam2)
  - requires initialization (mostly)
  - does have a video predictor
  - models can be loaded from the repo (seems tedious) or from hugging face (seems easier)
  - Demo testable online
  - SAM-Markers works better/worse with two markers, depending on marker placement
  - SAM-Everything (thing) segments areas pretty good, but does get shadowing behind object
    - does take a couple of seconds -> maybe speed up by pre-masking
  - SAM-Box (thing) does get the object well, but does get as much of the shadow behind as the box would allow
- [ ] [Watershed](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html)[^1] ?
  - requires setting markers
  - Test/watershed.py

[^1]: Requires "manual" initialization, i.e. the algorithms requires coordinates of the area, that should be segmented (or similar)
