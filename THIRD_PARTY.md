# Models and validation assets

The existing `face_landmarker.task` and `lbfmodel.yaml` files were retained. Their original download provenance was not recorded in this workspace; check the upstream model terms before redistribution.

- MediaPipe Face Landmarker API and models: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
- LBF model source referenced by the original application: https://github.com/kurnianggoro/GSOC2017/tree/master/data
- `haarcascade_frontalface_default.xml` was added from https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml because the local OpenCV installation lacked its data file. The upstream license notice is preserved inside the XML.
- Head-pose estimation uses OpenCV's solvePnP: https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html

Downloaded public validation images remain under the ignored `artifacts/` directory, are not runtime requirements, and are not bundled into the application:

- `mediapipe-portrait.jpg`: https://storage.googleapis.com/mediapipe-assets/portrait.jpg
- `opencv-face-sample.jpg`: https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg

The smoke video repeats a static portrait with black/dim frames to exercise calibration, tracking loss, quality checks, replay timing, and reports. It is not a driver dataset or evidence of real-world fatigue accuracy.

## Phone detection models

Downloaded from Google's official MediaPipe model storage for local inference:

- EfficientDet-Lite0 int8: https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
- Hand Landmarker: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
- Object detector documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/python
- Hand model documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python

`artifacts/phone-test.jpg` is **Woman Holding Up An iPhone.jpg** by **Xalgalgavon**, from https://commons.wikimedia.org/wiki/File:Woman_Holding_Up_An_iPhone.jpg, licensed under CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/. The downloaded original is unchanged; any validation previews crop/resize it and add detection overlays. This is a public test photo, not an image of the user or a driver. It is not a runtime asset.

Additional test photographs (also excluded from runtime):

- `phone-use-test.jpg`: **Woman holding cell phone 3236059.jpg**, Luisella Planeta, CC0, https://commons.wikimedia.org/wiki/File:Woman_holding_cell_phone_3236059.jpg. Its preview is resized.
- `phone-person-test.jpg`: **Man using smartphone in modern kitchen while smiling and enjoying the moment.jpg**, Nenad Stojković / Shixart1985, CC BY 2.0, https://commons.wikimedia.org/wiki/File:Man_using_smartphone_in_modern_kitchen_while_smiling_and_enjoying_the_moment.jpg. License: https://creativecommons.org/licenses/by/2.0/. `phone-person-test-preview.jpg` is resized; `phone-person-crop.jpg` is cropped to the upper body and resized to simulate closer camera framing. Validation dashboard images add detection overlays. Attribution applies to those derivatives as well.
