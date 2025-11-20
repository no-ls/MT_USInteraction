# NOTE
- quick build for testing different CV-Algorithms
  - OpenCV has existing [sample](https://github.com/opencv/opencv/tree/4.x/samples/python) implementations for graph cut (aka grabcut), optical flow and watershed 


## Usage
- install `cv2` and `numpy` (e.g. with pip)
- run: `py main.py -src path/to/png/or/mp4_or_png/file`
  - e.g: `py .\main.py -src ../Data/stressball.mp4`
- use number keys, e.g. <kbd>1</kbd> for threshold, to switch between algorithms
- use <kbd>arrow up</kbd> and <kbd>arrow down</kbd> to change the values used in the algorithm (e.g. threshold value)
- <kbd>q</kbd> or <kbd>esc</kbd> to quit
- <kbd>s</kbd> to save an image/frame

### Add Algorithms
- go to `Detection.py`
  - create new class, that inherits `Algorithm`
  - add the `def do_algorithm(self, value:int, img:MatLike, gray:MatLike):` function
  - in it return the image you want to see later
- got to the end of `Detection.py`
  - add/replace your class in the lists 
  - currently only detects number keys (0-9)


## Notes
- [ ] Try Denoising
  - see [Fast](https://fast-imaging.github.io/python-tutorial-ultrasound.html#autotoc_md159) for US image processing  
  - non local means filter
    - doesn't do much with recommended values, except tank fps
    - 20, 7, 21 ~ 1-2 fps
    - 20, 7, 7  ~ 15 fps
- [ ] Try mixing algorithms
  - [ ] Opening/Erode with color quant
  - [ ] Closing/Dilate with optical flow
  - [ ] Blur with color quant
- [x] Try different pre processing steps
  - [x] Low pass filter
  - [x] Different smoothing filters
  - [x] Morphological operations
    - Opening and Erode help with water speckle noise
    - Closing and Dilate amplify it
- [x] Different thresholds (adaptive, otsu)
  - -> not that great
  - Adaptive Thresholds
    - Mean seems noisier, than Gaussian
    - raw: bad -> too much noise, like little worms all over
    - Blur
      - (5,5)-Mean : barely a difference
      - (25,25)-Mean : better, but now very vague and torn apart, very hard to differentiate between noise and shape
  - Otsu
    - with pre-blur: bad -> very big "shapes", often pure white
    - without blur: similar, maybe a little better
  - Binary
    - better, given the correct values
- [x] thresholding (+ contour detection)
  - works without initialization
  - good fps ~20
- [x] k-means and/or fuzzy C-means
  - even with k=2 (you don't really need much more) very slow: fps ~6
  - gets contours pretty well
  - -> look into: color quantization
- [x] graph cuts[^1] ([grab cut](https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html))
  - requires initialization (does not work yet), can probably be adjusted live
  - bad performance -> would prob. require GPU use with CUDA (make with [install](https://coderivers.org/blog/install-opencv-python-with-cuda/))
  - Test with von OpenCv-Example
- [x] [optical flow](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html) ?
  - prob. good for application, where only the motion is important, not the detection -> e.g. Finger-Drawing
  - Test with von OpenCv-Example
- [ ] PDMs[^1] (deformable models) ?
- [x] existing medical DL Model
  - -> either wrong training data or requires "manual" initialization
  - e.g. U-Net
  - U-Net
    - pre-trained US model [here](https://github.com/izazahmad-ai/Ultrasound-Image-Segmentation-using-U-Net)
      - -> segments tumors and lesions -> objects would have to look like those
    - pre-trained model for high-definition images in PyTorch [here](https://github.com/milesial/Pytorch-UNet) 
      - trained on the Carvana dataset (Cars)
    - pre-trained models for lesion segmentation in US images [here](https://github.com/Yingping-LI/Light-U-net)
      - in folder `pretrained_models`
  - [x] Meta AI [segment anything](https://segment-anything.com/)[^1] aka [SAM](https://github.com/facebookresearch/segment-anything) or [SAM2](https://github.com/facebookresearch/sam2)
    - requires initialization (mostly)
    - does have a video predictor
    - models can be loaded from the repo (seems tedious) or from hugging face (seems easier)
    - Demo testable online
    - SAM-Markers works better/worse with two markers, depending on marker placement
    - SAM-Everything (thing) segments areas pretty good, but does get shadowing behind object
      - does take a couple of seconds -> maybe speed up by pre-masking
    - SAM-Box (thing) does get the object well, but does get as much of the shadow behind as the box would allow
    - SAM2 can do video segmentation with box or marker
- [x] [Watershed](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html)[^1] ?
  - requires setting markers
  - Test with von OpenCv-Example

[^1]: Requires "manual" initialization, i.e. the algorithms requires coordinates of the area, that should be segmented (or similar)
