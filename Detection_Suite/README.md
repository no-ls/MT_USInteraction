### Test Data
- [ ] Materials to test
  - [ ] Aluminium
  - [ ] Agar Agar phantom
  - [ ] Wood/Deformable Wax
  - [ ] Stressball
  - [ ] Plastic
- [ ] get 2-5 different **images** of the materials
- [ ] get 1-2 **video** clips of the materials including vertical, horizontal and diagonal movement (+ taking out/putting in)
- [ ] ? get images of foreign bodies from existing datasets ?

### Algorithms to Compare
- [x] thresholding
- [ ] k-means and/or fuzzy C-means (an MA Projekt orientieren)
- [ ] graph cuts
  - Requires "manual" initialization
- [ ] optical flow ?
- [ ] PDMs (deformable models) ?
  - Requires "manual" initialization
- [ ] Contour detection (not active contours, too similar to PDMs) ?
- [ ] existing medical DL Model
- [ ] Meta AI [segment anything](https://segment-anything.com/)
  - Requires "manual" initialization

"manual" initialization means that the algorithms requires coordinates of the area, that should be segmented (or similar)
