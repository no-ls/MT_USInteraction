# MT_USInteraction
Code of Masters Thesis on exploring the potential of medical ultrasound imaging for HCI.

For running the demos see the Demo-[README](./Demos/README.md)

## Notes

#### ...on Segmentation:
The demos were implemented with traditional CV-based segmentation methods: (adaptive) threshold and color quantization; to keep things as simple, and computationally cheap as possible.
This approach was chosen as current DL-models (as of spring 2026) either require manual initialization anyway (like SAM2), or require a specific data structures, and/or positional data (many of the 3D US models).
For simple applications, especially those using hyperechoic (bright) object, these approaches work very well for the water bath setup.
Other materials can be harder to separate from noise fully.

#### ...on Materials:
Good materials to use with the water bath setup:
- fingers
- ultrasound phantoms (e.g. agar)
- 3D print (with a mesh like structure) (see: `Data/Images/scanbed.jpg` (left image))
  - plastic typically fully reflects sound waves, therefore small gaps are needed for a full cross-section
- gel-filled stressball
- aluminum foil (good visibility, but creates artifacts (bright shadowing))

Other materials, that can be used, but mostly show the front edge:
- wood
- plastic
- modeling wax
- glass
- acrylic sheet (can be used as a mirror to reflect sound waves)